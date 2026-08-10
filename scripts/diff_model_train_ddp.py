# Fine-tune the MAISI rflow diffusion UNet on pre-extracted latents.
#
# Stage 2 only: the VAE is frozen and is loaded on rank 0 solely for the img2img
# visual check. The UNet is the only module in the optimizer.
#
#   torchrun --nproc_per_node=8 -m scripts.diff_model_train_ddp \
#       -t ./configs/config_network_rflow.json \
#       -c ./configs/config_maisi_diff_model_rflow-mr.json \
#       -e ./configs/environment_maisi_diff_model_rflow-mr.json -g 8

from __future__ import annotations

import argparse
import csv
import json
import os

import monai
import numpy as np
import torch
import torch.distributed as dist
from monai.data import DataLoader
from monai.inferers.inferer import SlidingWindowInferer
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import Compose
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from .diff_model_setting import initialize_distributed, load_config, setup_logging
from .diff_model_train import augment_modality_label
from .utils import define_instance, dynamic_infer
from .utils_infer import ReconModel
from .utils_plot import get_xyz_plot

# Loss is a single L1, but split by noise level: rflow error is very unevenly
# distributed over tau, and the mean hides a stalled band.
LOSS_KEYS = ("l1", "l1_lo", "l1_mid", "l1_hi")
TEST_LOSS_KEYS = ("l1", "l1_lo", "l1_mid", "l1_hi")
TRAIN_LOG_HEADER = (
    ["epoch", "lr"] + [f"train_{k}" for k in LOSS_KEYS] + [f"test_{k}" for k in TEST_LOSS_KEYS] + ["n_train", "n_test"]
)

# Fixed (t, eps) for the test split so the number is comparable across epochs.
TEST_SEED = 20260101
IMG2IMG_SEED = 20260202


def append_csv(path, row, header):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def reduce_sum(value, device):
    t = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def _load_json_field(path, key, as_float=True):
    with open(path) as f:
        d = json.load(f)
    return torch.FloatTensor(d[key]) if as_float else d[key]


def build_transforms(modality_mapping):
    return Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: _load_json_field(x, "spacing")),
            monai.transforms.Lambdad(keys="spacing", func=lambda x: x * 1e2),
            monai.transforms.Lambdad(keys="modality", func=lambda x: modality_mapping[_load_json_field(x, "modality", False)]),
            monai.transforms.EnsureTyped(keys=["modality"], dtype=torch.long),
        ]
    )


def build_file_list(datalist_path, embedding_base_dir):
    """dataset.json image paths -> the matching *_emb.nii.gz + sidecar json."""
    with open(datalist_path) as f:
        items = json.load(f)["training"]

    files, missing_emb, missing_json = [], 0, 0
    for item in items:
        rel = item["image"].replace(".nii.gz", "_emb.nii.gz").lstrip("/")
        emb = os.path.join(embedding_base_dir, rel)
        if not os.path.exists(emb):
            missing_emb += 1
            continue
        info = emb + ".json"
        if not os.path.exists(info):
            missing_json += 1
            continue
        files.append({"image": emb, "spacing": info, "modality": info})
    return files, missing_emb, missing_json


class RankSliceSampler(Sampler):
    """range(rank, N, world) -- exact partition, no padding.

    DistributedSampler pads to equalise rank lengths, which double-counts test
    samples. The test loop runs no collectives, so unequal per-rank step counts
    are safe here.
    """

    def __init__(self, n, rank, world_size):
        self.indices = list(range(rank, n, world_size))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


class DiffusionTrainerDDP:
    def __init__(self, args, num_gpus, amp=True):
        self.args = args
        self.amp = amp
        self.train_cfg = args.diffusion_unet_train

        self.local_rank, self.world_size, self.device = initialize_distributed(num_gpus)
        self.logger = setup_logging("train_ddp")
        self.is_main = self.local_rank == 0

        self.log_dir = getattr(args, "train_log_path", os.path.join(args.model_dir, "logs"))
        self.train_log_csv = os.path.join(self.log_dir, "train_log.csv")
        self.img_dir = os.path.join(self.log_dir, "img2img")
        if self.is_main:
            for d in (args.model_dir, self.log_dir, self.img_dir):
                os.makedirs(d, exist_ok=True)
        if dist.is_initialized():
            dist.barrier()

        self.num_train_timesteps = args.noise_scheduler["num_train_timesteps"]
        self.grad_clip = self.train_cfg.get("grad_clip_norm", 1.0)
        self.modality_dropout = self.train_cfg.get("modality_dropout_prob", 0.1)
        self.test_interval = self.train_cfg.get("test_epoch_interval", 1)
        self.img2img_interval = self.train_cfg.get("img2img_epoch_interval", 5)
        self.ckpt_interval = self.train_cfg.get("ckpt_save_epoch_interval", 1)
        self.n_epochs = self.train_cfg["n_epochs"]

        with open(args.modality_mapping_path) as f:
            self.modality_mapping = json.load(f)

        self.noise_scheduler = define_instance(args, "noise_scheduler")
        assert isinstance(self.noise_scheduler, RFlowScheduler), "this trainer targets the rflow variant"

        self._build_data()
        self._build_test_data()
        self._build_model()
        self._build_optim()

        self.autoencoder = None  # lazily built on rank 0 for img2img
        self.start_epoch = 0
        if self.train_cfg.get("if_resume", False):
            self._resume(os.path.join(args.model_dir, args.model_filename))

    # ------------------------------------------------------------------ data
    def _build_data(self):
        files, miss_emb, miss_json = build_file_list(self.args.json_data_list, self.args.embedding_base_dir)
        if not files:
            raise RuntimeError(f"no usable embeddings under {self.args.embedding_base_dir}")
        self.n_train = len(files)
        if self.is_main:
            self.logger.info(f"[data] train {len(files)} usable | missing emb {miss_emb} | missing json {miss_json}")

        ds = monai.data.Dataset(data=files, transform=build_transforms(self.modality_mapping))
        bs = max(1, self.train_cfg["batch_size"] // self.world_size)
        nw = self.train_cfg.get("num_workers", 8)
        self.train_sampler = DistributedSampler(ds, self.world_size, self.local_rank, shuffle=True, drop_last=True)
        self.train_loader = DataLoader(
            ds, batch_size=bs, num_workers=nw, drop_last=True, sampler=self.train_sampler,
            pin_memory=True, prefetch_factor=4 if nw > 0 else None,
        )
        self.per_gpu_bs = bs
        if self.is_main:
            self.logger.info(f"[data] per-GPU bs {bs} | {len(self.train_loader)} steps/epoch/rank")

    def _build_test_data(self):
        self.test_loader = None
        self.test_files = []
        self.n_test = 0

        base = getattr(self.args, "test_embedding_base_dir", None)
        dl = getattr(self.args, "test_json_data_list", None)
        if not base or not dl:
            if self.is_main:
                self.logger.info("[data] test_embedding_base_dir / test_json_data_list missing -> test disabled")
            return

        files, miss_emb, miss_json = build_file_list(dl, base)
        cap = self.train_cfg.get("test_max_cases", 500)
        files = files[:cap]  # deterministic prefix: same cases every epoch
        if not files:
            if self.is_main:
                self.logger.info("[data] test set empty -> test disabled")
            return

        self.test_files = files
        self.n_test = len(files)
        ds = monai.data.Dataset(data=files, transform=build_transforms(self.modality_mapping))
        bs = max(1, self.train_cfg.get("test_batch_size", self.train_cfg["batch_size"]) // self.world_size)
        nw = self.train_cfg.get("num_workers", 8)
        self.test_loader = DataLoader(
            ds, batch_size=bs, num_workers=nw, drop_last=False,
            sampler=RankSliceSampler(len(ds), self.local_rank, self.world_size),
            pin_memory=True, prefetch_factor=4 if nw > 0 else None,
        )
        if self.is_main:
            self.logger.info(f"[data] test {self.n_test} cases (cap {cap}) | missing emb {miss_emb} | missing json {miss_json}")

    # ----------------------------------------------------------------- model
    def _build_model(self):
        unet = define_instance(self.args, "diffusion_unet_def").to(self.device)

        ckpt_path = getattr(self.args, "existing_ckpt_filepath", None)
        if ckpt_path is None:
            self.scale_factor = None
            self.logger.info("[model] existing_ckpt_filepath is null -> training from scratch")
        else:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            missing, unexpected = unet.load_state_dict(ckpt["unet_state_dict"], strict=False)
            self.scale_factor = ckpt["scale_factor"]
            self.logger.info(f"[model] loaded {ckpt_path}")
            self.logger.info(f"[model] missing={list(missing)} unexpected={list(unexpected)}")
            self.logger.info(f"[model] scale_factor from ckpt -> {float(self.scale_factor):.6f}")

        if self.scale_factor is None:
            batch = next(iter(self.train_loader))
            sf = 1.0 / torch.std(batch["image"].to(self.device))
            if dist.is_initialized():
                dist.all_reduce(sf, op=dist.ReduceOp.AVG)
            self.scale_factor = sf
            self.logger.info(f"[model] scale_factor computed from data -> {float(sf):.6f}")

        self.unet_module = unet
        if dist.is_initialized():
            self.unet = DistributedDataParallel(unet, device_ids=[self.device], find_unused_parameters=True)
            self.unet_module = self.unet.module
        else:
            self.unet = unet

        n_param = sum(p.numel() for p in self.unet_module.parameters() if p.requires_grad)
        if self.is_main:
            self.logger.info(f"[model] trainable params {n_param / 1e6:.2f} M (UNet only; VAE and scheduler have none)")

    def _build_optim(self):
        self.optimizer = torch.optim.Adam(self.unet.parameters(), lr=self.train_cfg["lr"])
        total_steps = self.n_epochs * len(self.train_loader)
        self.lr_scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=total_steps, power=2.0)
        self.scaler = GradScaler("cuda")

    def _resume(self, path):
        if not os.path.exists(path):
            if self.is_main:
                self.logger.info(f"[resume] {path} not found -> starting fresh")
            return
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.unet_module.load_state_dict(ck["unet_state_dict"])
        if "optimizer_state_dict" in ck:
            self.optimizer.load_state_dict(ck["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(ck["lr_scheduler_state_dict"])
            self.scaler.load_state_dict(ck["scaler_state_dict"])
        self.scale_factor = ck["scale_factor"]
        self.start_epoch = ck["epoch"]
        if self.is_main:
            self.logger.info(f"[resume] from epoch {self.start_epoch}")

    # ------------------------------------------------------------------ step
    def _forward(self, batch, net, train_mode):
        """-> per-sample L1 (B,) and tau (B,) in [0,1], 0 = clean, 1 = pure noise."""
        z = batch["image"].to(self.device, non_blocking=True) * self.scale_factor
        spacing = batch["spacing"].to(self.device)
        modality = batch["modality"].to(self.device)
        if train_mode:
            modality = augment_modality_label(modality, prob=self.modality_dropout)

        with autocast("cuda", enabled=self.amp):
            noise = torch.randn_like(z)
            timesteps = self.noise_scheduler.sample_timesteps(z)
            x_t = self.noise_scheduler.add_noise(original_samples=z, noise=noise, timesteps=timesteps)
            v_hat = net(x=x_t, timesteps=timesteps, spacing_tensor=spacing, class_labels=modality)
            v = z - noise
            per_sample = (v_hat.float() - v.float()).abs().flatten(1).mean(1)

        tau = timesteps.float().to(self.device) / self.num_train_timesteps
        return per_sample, tau

    @staticmethod
    def _accumulate(bucket, per_sample, tau):
        d = per_sample.detach().float()
        bucket["l1_sum"] += float(d.sum())
        bucket["l1_n"] += d.numel()
        for name, sel in (("lo", tau < 1 / 3), ("mid", (tau >= 1 / 3) & (tau < 2 / 3)), ("hi", tau >= 2 / 3)):
            if sel.any():
                bucket[f"{name}_sum"] += float(d[sel].sum())
                bucket[f"{name}_n"] += int(sel.sum())

    def _finalize(self, bucket):
        out = {}
        for key, prefix in (("l1", "l1"), ("l1_lo", "lo"), ("l1_mid", "mid"), ("l1_hi", "hi")):
            s = reduce_sum(bucket[f"{prefix}_sum"], self.device)
            n = reduce_sum(bucket[f"{prefix}_n"], self.device)
            out[key] = s / n if n > 0 else -1.0
        return out

    @staticmethod
    def _empty_bucket():
        return {f"{p}_{s}": 0.0 for p in ("l1", "lo", "mid", "hi") for s in ("sum", "n")}

    # ----------------------------------------------------------------- train
    def train_epoch(self, epoch):
        self.unet.train()
        self.train_sampler.set_epoch(epoch)
        bucket = self._empty_bucket()

        pbar = tqdm(self.train_loader, desc=f"Train ep{epoch}", disable=not self.is_main)
        for it, batch in enumerate(pbar):
            per_sample, tau = self._forward(batch, self.unet, train_mode=True)
            loss = per_sample.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at ep{epoch} it{it}")

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.unet.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.lr_scheduler.step()

            self._accumulate(bucket, per_sample, tau)
            if self.is_main:
                pbar.set_postfix(l1=f"{loss.item():.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")

        return self._finalize(bucket)

    @torch.no_grad()
    def test_epoch(self, epoch):
        """Forward through unet_module (NOT the DDP wrapper): RankSliceSampler gives
        ranks unequal step counts, so the loop must contain no collectives."""
        if self.test_loader is None:
            return {k: -1.0 for k in TEST_LOSS_KEYS}

        self.unet.eval()
        # Same seed every epoch + fixed sample order => same (t, eps) per case.
        torch.manual_seed(TEST_SEED + self.local_rank)
        torch.cuda.manual_seed_all(TEST_SEED + self.local_rank)

        bucket = self._empty_bucket()
        for batch in tqdm(self.test_loader, desc=f"Test  ep{epoch}", disable=not self.is_main):
            per_sample, tau = self._forward(batch, self.unet_module, train_mode=False)
            self._accumulate(bucket, per_sample, tau)
        return self._finalize(bucket)

    # --------------------------------------------------------------- img2img
    def _build_autoencoder(self):
        ae = define_instance(self.args, "autoencoder_def").to(self.device)
        ck = torch.load(self.args.trained_autoencoder_path, map_location=self.device, weights_only=False)
        if "unet_state_dict" in ck:
            ck = ck["unet_state_dict"]
        ae.load_state_dict(ck)
        ae.eval()
        self.autoencoder = ae
        self.recon_model = ReconModel(autoencoder=ae, scale_factor=self.scale_factor).to(self.device)
        self.recon_inferer = SlidingWindowInferer(
            roi_size=[80, 80, 80], sw_batch_size=1, progress=False, mode="gaussian",
            overlap=0.4, sw_device=self.device, device=self.device,
        )

    @torch.no_grad()
    def img2img(self, epoch):
        """Real latent -> noise to tau0 in (0, tau_max] -> integrate back -> decode.
        The only view that puts a generated volume next to its real counterpart."""
        if not self.is_main or not self.test_files:
            return
        if self.autoencoder is None:
            self._build_autoencoder()

        n_cases = self.train_cfg.get("img2img_num_cases", 8)
        tau_max = self.train_cfg.get("img2img_tau_max", 0.5)
        n_steps = self.train_cfg.get("img2img_num_inference_steps", 30)
        cfg_scale = self.args.diffusion_unet_inference.get("cfg_guidance_scale", 0.0)
        modality_code = self.args.diffusion_unet_inference["modality"]

        transform = build_transforms(self.modality_mapping)
        self.unet.eval()
        gen = torch.Generator().manual_seed(IMG2IMG_SEED)
        panels = []

        for case in self.test_files[:n_cases]:
            item = transform(dict(case))
            z = item["image"][None].to(self.device).float() * self.scale_factor
            spacing = item["spacing"][None].to(self.device).float()
            modality = torch.tensor([modality_code], dtype=torch.long, device=self.device)

            tau0 = float(torch.rand(1, generator=gen)) * tau_max
            t0 = tau0 * self.num_train_timesteps
            noise = torch.randn(z.shape, generator=gen).to(self.device)
            x = self.noise_scheduler.add_noise(
                original_samples=z, noise=noise,
                timesteps=torch.tensor([t0], device=self.device),
            )

            self.noise_scheduler.set_timesteps(
                num_inference_steps=n_steps,
                input_img_size_numel=torch.prod(torch.tensor(z.shape[2:])),
            )
            ts = self.noise_scheduler.timesteps
            ts = torch.cat((torch.tensor([t0], dtype=ts.dtype), ts[ts < t0]))
            next_ts = torch.cat((ts[1:], torch.tensor([0], dtype=ts.dtype)))

            with autocast("cuda", enabled=True):
                for t, next_t in zip(ts, next_ts):
                    inputs = {
                        "x": x,
                        "timesteps": torch.Tensor((t,)).to(self.device),
                        "spacing_tensor": spacing,
                        "class_labels": modality,
                    }
                    if cfg_scale > 0:
                        for k in inputs:
                            if k == "class_labels":
                                inputs[k] = torch.cat([inputs[k], torch.zeros_like(modality)])
                            else:
                                inputs[k] = torch.cat([inputs[k]] * 2)
                        cond, uncond = self.unet_module(**inputs).chunk(2)
                        v_hat = uncond + cfg_scale * (cond - uncond)
                    else:
                        v_hat = self.unet_module(**inputs)
                    x, _ = self.noise_scheduler.step(v_hat, t, x, next_t)

                real = dynamic_infer(self.recon_inferer, self.recon_model, z)
                fake = dynamic_infer(self.recon_inferer, self.recon_model, x)

            centers = [s // 2 for s in real.shape[2:]]
            pair = np.concatenate(
                [
                    get_xyz_plot(real[0].float().cpu(), centers, mask_bool=False),
                    get_xyz_plot(fake[0].float().cpu(), centers, mask_bool=False),
                ],
                axis=0,
            )
            panels.append((pair, tau0))

        from PIL import Image

        grid = np.concatenate([p for p, _ in panels], axis=1)
        lo, hi = np.percentile(grid, 0.5), np.percentile(grid, 99.5)
        grid = np.clip((grid - lo) / max(hi - lo, 1e-6), 0, 1)
        out = os.path.join(self.img_dir, f"img2img_epoch_{epoch:04d}.png")
        Image.fromarray((grid * 255).astype(np.uint8)).save(out)
        taus = ", ".join(f"{t:.2f}" for _, t in panels)
        self.logger.info(f"[img2img] saved {out} | top row = real, bottom row = regenerated | tau0 = [{taus}]")

    # ------------------------------------------------------------------ loop
    def save_checkpoint(self, epoch, train_loss):
        path = os.path.join(self.args.model_dir, self.args.model_filename)
        torch.save(
            {
                "epoch": epoch + 1,
                "loss": train_loss["l1"],
                "num_train_timesteps": self.num_train_timesteps,
                "scale_factor": self.scale_factor,
                "unet_state_dict": self.unet_module.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            },
            path,
        )
        self.logger.info(f"[ckpt] saved {path}")

    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs):
            tr = self.train_epoch(epoch)

            do_test = (epoch % self.test_interval == 0) or (epoch == self.n_epochs - 1)
            te = self.test_epoch(epoch) if do_test else {k: -1.0 for k in TEST_LOSS_KEYS}

            if (epoch % self.img2img_interval == 0) or (epoch == self.n_epochs - 1):
                self.img2img(epoch)

            if self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                append_csv(
                    self.train_log_csv,
                    [epoch, lr] + [tr[k] for k in LOSS_KEYS] + [te[k] for k in TEST_LOSS_KEYS] + [self.n_train, self.n_test],
                    TRAIN_LOG_HEADER,
                )
                self.logger.info(
                    f"ep{epoch} lr={lr:.3e} | TRAIN l1={tr['l1']:.4f} "
                    f"(lo={tr['l1_lo']:.4f} mid={tr['l1_mid']:.4f} hi={tr['l1_hi']:.4f}) | "
                    f"TEST l1={te['l1']:.4f} (lo={te['l1_lo']:.4f} mid={te['l1_mid']:.4f} hi={te['l1_hi']:.4f})"
                )

                if (epoch % self.ckpt_interval == 0) or (epoch == self.n_epochs - 1):
                    self.save_checkpoint(epoch, tr)

            if dist.is_initialized():
                dist.barrier()

        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="MAISI rflow diffusion UNet fine-tuning (DDP)")
    parser.add_argument("-e", "--env_config_path", type=str, required=True)
    parser.add_argument("-c", "--model_config_path", type=str, required=True)
    parser.add_argument("-t", "--model_def_path", type=str, required=True)
    parser.add_argument("-g", "--num_gpus", type=int, default=1)
    parser.add_argument("--no_amp", dest="amp", action="store_false")
    cli = parser.parse_args()

    args = load_config(cli.env_config_path, cli.model_config_path, cli.model_def_path)
    DiffusionTrainerDDP(args, cli.num_gpus, amp=cli.amp).train()


if __name__ == "__main__":
    main()
