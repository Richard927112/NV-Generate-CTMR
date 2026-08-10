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
import nibabel as nib
import numpy as np
import torch
import torch.distributed as dist
from monai.data import DataLoader
from monai.inferers.inferer import SlidingWindowInferer
from monai.metrics.fid import FIDMetric
from monai.networks.schedulers import RFlowScheduler
from monai.transforms import Compose
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
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
    ["epoch", "lr"]
    + [f"train_{k}" for k in LOSS_KEYS]
    + [f"test_{k}" for k in TEST_LOSS_KEYS]
    + ["i2i_lat_rel", "i2i_mse", "i2i_psnr", "i2i_ssim"]
    + ["dist_fd", "dist_w1"]
    + ["n_train", "n_test"]
)
IMG2IMG_HEADER = ["epoch", "case_id", "tau0", "lat_l2_rel", "mse", "psnr", "ssim"]

# Fixed (t, eps) for the test split so the number is comparable across epochs.
TEST_SEED = 20260101
IMG2IMG_SEED = 20260202
DIST_SEED = 20260303


def append_csv(path, row, header):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow(row)


def append_rows_csv(path, rows, header):
    if not rows:
        return
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerows(rows)


def reduce_sum(value, device):
    t = torch.tensor(float(value), dtype=torch.float64, device=device)
    if dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


def build_transforms():
    """Latents only. spacing and modality are constants here (see DiffusionTrainerDDP)."""
    return Compose(
        [
            monai.transforms.LoadImaged(keys=["image"]),
            monai.transforms.EnsureChannelFirstd(keys=["image"]),
        ]
    )


def build_file_list(datalist_path, embedding_base_dir, subdir, json_key):
    """dataset json -> <embedding_base_dir>/<subdir>/<accession>/<stem>_emb.nii.gz

    The accession folder is the LAST directory of the original image path:
      /mnt/.../FDZL/20260409/11193855_20200417_02200417199019/T2WI_AX_1.nii.gz
                             ^^^^^^^^^^ accession ^^^^^^^^^^  ^^ stem ^^
    """
    with open(datalist_path) as f:
        items = json.load(f)[json_key]

    files, missing, tried = [], 0, []
    for item in items:
        src = item["image"]
        accession = os.path.basename(os.path.dirname(src))
        stem = os.path.basename(src).replace(".gz", "").replace(".nii", "")
        emb = os.path.join(embedding_base_dir, subdir, accession, stem + "_emb.nii.gz")
        if os.path.exists(emb):
            files.append({"image": emb})
        else:
            missing += 1
            if len(tried) < 5:
                tried.append(emb)
    return files, missing, tried


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
        self.img2img_csv = os.path.join(self.log_dir, "img2img_metrics.csv")
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
        self.dist_interval = self.train_cfg.get("dist_epoch_interval", 5)
        self.ckpt_interval = self.train_cfg.get("ckpt_save_epoch_interval", 1)
        self.n_epochs = self.train_cfg["n_epochs"]

        # Every latent was resampled to the same grid during extraction, so spacing is a
        # constant, not per-case metadata; modality is a constant too (single-sequence
        # dataset). No sidecar json is read anywhere.
        infer_cfg = args.diffusion_unet_inference
        self.spacing_tensor = (torch.tensor(infer_cfg["spacing"], dtype=torch.float32) * 1e2).to(self.device)
        self.modality_code = int(infer_cfg["modality"])
        if self.is_main:
            self.logger.info(
                f"[cond] spacing={infer_cfg['spacing']} (x100 -> {self.spacing_tensor.tolist()}) "
                f"| modality={self.modality_code} -- must match what extraction used"
            )

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
        files, missing, tried = build_file_list(
            self.args.json_data_list,
            self.args.embedding_base_dir,
            getattr(self.args, "train_embedding_subdir", "train_data"),
            getattr(self.args, "train_json_key", "training"),
        )
        if not files:
            raise RuntimeError(
                f"no embeddings resolved under {self.args.embedding_base_dir}. Tried e.g.:\n  " + "\n  ".join(tried)
            )
        self.n_train = len(files)
        if self.is_main:
            self.logger.info(f"[data] train {len(files)} resolved | {missing} not found")
            if missing:
                self.logger.info(f"[data] first miss -> {tried[0]}")

        ds = monai.data.Dataset(data=files, transform=build_transforms())
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

        files, missing, tried = build_file_list(
            dl, base,
            getattr(self.args, "test_embedding_subdir", "test_data"),
            getattr(self.args, "test_json_key", "testing"),
        )
        cap = self.train_cfg.get("test_max_cases", 500)
        files = files[:cap]  # deterministic prefix: same cases every epoch
        if not files:
            if self.is_main:
                self.logger.info("[data] test set empty -> test disabled")
                if tried:
                    self.logger.info(f"[data] tried e.g. {tried[0]}")
            return

        self.test_files = files
        self.n_test = len(files)
        ds = monai.data.Dataset(data=files, transform=build_transforms())
        bs = max(1, self.train_cfg.get("test_batch_size", self.train_cfg["batch_size"]) // self.world_size)
        nw = self.train_cfg.get("num_workers", 8)
        self.test_loader = DataLoader(
            ds, batch_size=bs, num_workers=nw, drop_last=False,
            sampler=RankSliceSampler(len(ds), self.local_rank, self.world_size),
            pin_memory=True, prefetch_factor=4 if nw > 0 else None,
        )
        if self.is_main:
            self.logger.info(f"[data] test {self.n_test} cases (cap {cap}) | {missing} not found")

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
        b = z.shape[0]
        spacing = self.spacing_tensor[None].repeat(b, 1)
        modality = torch.full((b,), self.modality_code, dtype=torch.long, device=self.device)
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
    def _denoise_from(self, z, noise, tau0, spacing, modality, n_steps, cfg_scale):
        """Start at x = (1-tau0)*z + tau0*eps and integrate the ODE back to tau=0.

        tau0 = 1.0 gives x = eps, i.e. ordinary generation from pure noise.
        Batch-safe: z may carry any batch size B.
        """
        batch = z.shape[0]
        t0 = tau0 * self.num_train_timesteps
        x = self.noise_scheduler.add_noise(
            original_samples=z, noise=noise, timesteps=torch.tensor([t0], device=self.device)
        )
        self.noise_scheduler.set_timesteps(
            num_inference_steps=n_steps, input_img_size_numel=torch.prod(torch.tensor(z.shape[2:]))
        )
        ts = self.noise_scheduler.timesteps
        ts = torch.cat((torch.tensor([t0], dtype=ts.dtype), ts[ts < t0]))
        next_ts = torch.cat((ts[1:], torch.tensor([0], dtype=ts.dtype)))

        with autocast("cuda", enabled=True):
            for t, next_t in zip(ts, next_ts):
                inputs = {
                    "x": x,
                    "timesteps": torch.full((batch,), float(t), device=self.device),
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
        return x

    @staticmethod
    def _montage(volume, lo, hi):
        """(1,1,H,W,D) tensor -> uint8 3-view montage, windowed by the real volume's range."""
        centers = [s // 2 for s in volume.shape[2:]]
        img = get_xyz_plot(volume[0].float().cpu(), centers, mask_bool=False)
        img = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
        return (img * 255).astype(np.uint8)

    @torch.no_grad()
    def img2img(self, epoch):
        """Real latent -> partial noise -> integrate back -> decode -> compare with the real one.

        tau0 is a FIXED list, and the gaussian per case is seeded, so every epoch runs the
        exact same experiment and the numbers form a curve. Higher psnr is NOT strictly
        better: a model that returned the input untouched would score perfectly.
        """
        if not self.is_main or not self.test_files:
            return None
        if self.autoencoder is None:
            self._build_autoencoder()

        n_cases = self.train_cfg.get("img2img_num_cases", 8)
        taus = self.train_cfg.get("img2img_taus", [0.2, 0.35, 0.5])
        n_steps = self.train_cfg.get("img2img_num_inference_steps", 30)
        save_nifti = self.train_cfg.get("img2img_save_nifti", True)
        cfg_scale = self.args.diffusion_unet_inference.get("cfg_guidance_scale", 0.0)

        transform = build_transforms()
        self.unet.eval()
        nii_dir = os.path.join(self.img_dir, f"epoch_{epoch:04d}")
        if save_nifti:
            os.makedirs(nii_dir, exist_ok=True)

        rows, records, panels = [], [], []
        for ci, case in enumerate(self.test_files[:n_cases]):
            case_id = os.path.basename(case["image"]).replace("_emb.nii.gz", "")
            item = transform(dict(case))
            z = item["image"][None].to(self.device).float() * self.scale_factor
            spacing = self.spacing_tensor[None]
            modality = torch.tensor([self.modality_code], dtype=torch.long, device=self.device)

            # Same gaussian for every tau of this case: the tau sweep is then a clean ablation.
            gen = torch.Generator().manual_seed(IMG2IMG_SEED + ci)
            noise = torch.randn(z.shape, generator=gen).to(self.device)

            with autocast("cuda", enabled=True):
                real = dynamic_infer(self.recon_inferer, self.recon_model, z)
            real_np = real.squeeze().float().cpu().numpy()
            lo, hi = float(np.percentile(real_np, 0.5)), float(np.percentile(real_np, 99.5))
            data_range = float(real_np.max() - real_np.min())
            row_imgs = [self._montage(real, lo, hi)]

            if save_nifti:
                affine = np.diag(list(self.args.diffusion_unet_inference["spacing"]) + [1.0])
                nib.save(nib.Nifti1Image(real_np, affine), os.path.join(nii_dir, f"{case_id}_real.nii.gz"))

            for tau0 in taus:
                x = self._denoise_from(z, noise, float(tau0), spacing, modality, n_steps, cfg_scale)
                with autocast("cuda", enabled=True):
                    fake = dynamic_infer(self.recon_inferer, self.recon_model, x)
                fake_np = fake.squeeze().float().cpu().numpy()

                lat_rel = float(torch.linalg.vector_norm(x - z) / torch.linalg.vector_norm(z))
                mse = float(np.mean((real_np - fake_np) ** 2))
                psnr = float(peak_signal_noise_ratio(real_np, fake_np, data_range=data_range))
                ssim = float(structural_similarity(real_np, fake_np, data_range=data_range))

                rows.append([epoch, case_id, round(float(tau0), 3), lat_rel, mse, psnr, ssim])
                records.append(
                    {"case_id": case_id, "tau0": float(tau0), "lat_l2_rel": lat_rel, "mse": mse, "psnr": psnr, "ssim": ssim}
                )
                row_imgs.append(self._montage(fake, lo, hi))
                if save_nifti:
                    nib.save(nib.Nifti1Image(fake_np, affine), os.path.join(nii_dir, f"{case_id}_tau{tau0:.2f}.nii.gz"))

            panels.append(np.concatenate(row_imgs, axis=1))

        # one PNG: rows = cases, columns = [real | tau_1 | tau_2 | ...]
        png = os.path.join(self.img_dir, f"img2img_epoch_{epoch:04d}.png")
        Image.fromarray(np.concatenate(panels, axis=0)).save(png)

        append_rows_csv(self.img2img_csv, rows, IMG2IMG_HEADER)
        summary = {
            "epoch": epoch,
            "taus": [float(t) for t in taus],
            "num_cases": len(panels),
            "num_inference_steps": n_steps,
            "cfg_guidance_scale": cfg_scale,
            "per_tau": {
                f"{t:.2f}": {
                    m: float(np.mean([r[m] for r in records if r["tau0"] == float(t)]))
                    for m in ("lat_l2_rel", "mse", "psnr", "ssim")
                }
                for t in taus
            },
            "per_case": records,
        }
        with open(os.path.join(self.img_dir, f"img2img_epoch_{epoch:04d}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        worst = summary["per_tau"][f"{max(taus):.2f}"]
        self.logger.info(
            f"[img2img] {png} | columns = real, " + ", ".join(f"tau={t:.2f}" for t in taus) + " | "
            f"at tau={max(taus):.2f}: lat_rel={worst['lat_l2_rel']:.4f} mse={worst['mse']:.5f} "
            f"psnr={worst['psnr']:.2f} ssim={worst['ssim']:.4f}"
        )
        return worst

    # --------------------------------------------------------- distribution
    def _real_latents(self, n):
        """First n real test latents, scaled the same way training scales them."""
        transform = build_transforms()
        out = [transform(dict(c))["image"][None].float() * float(self.scale_factor) for c in self.test_files[:n]]
        return torch.cat(out, dim=0)

    @staticmethod
    def _pooled_features(latents):
        """(N,C,D,H,W) -> (N, C*8): spatial average-pool to 2x2x2 then flatten.

        Keeps the feature dim (32 for C=4) well below the sample count so the
        covariance in the Frechet distance stays conditioned.
        """
        return torch.nn.functional.adaptive_avg_pool3d(latents, (2, 2, 2)).flatten(1)

    @staticmethod
    def _channel_w1(a, b, n_values=20000, seed=0):
        """Mean per-channel 1-D Wasserstein distance between two sets of latents.

        For equal-size samples W1 is just the mean absolute gap between the two
        sorted value vectors, so no scipy and no histogram binning is needed.
        """
        g = torch.Generator().manual_seed(seed)
        dists = []
        for c in range(a.shape[1]):
            va, vb = a[:, c].flatten(), b[:, c].flatten()
            k = min(n_values, va.numel(), vb.numel())
            ia = torch.randperm(va.numel(), generator=g)[:k]
            ib = torch.randperm(vb.numel(), generator=g)[:k]
            dists.append(float((va[ia].sort().values - vb[ib].sort().values).abs().mean()))
        return float(np.mean(dists))

    @torch.no_grad()
    def dist_check(self, epoch):
        """Generate N latents from pure noise and compare their distribution to the
        real test latents.

        NOT FID: a real FID needs an image-domain pretrained feature extractor. This
        is a Frechet distance on pooled VAE-latent features plus a per-channel
        Wasserstein-1. It is a failure detector (mode collapse, intensity drift,
        noise-like output), not a quality score.
        """
        if not self.is_main or not self.test_files:
            return None

        n_samples = self.train_cfg.get("dist_num_samples", 128)
        n_samples = min(n_samples, len(self.test_files))
        batch = self.train_cfg.get("dist_batch_size", 8)
        n_steps = self.train_cfg.get("dist_num_inference_steps", 30)
        cfg_scale = self.args.diffusion_unet_inference.get("cfg_guidance_scale", 0.0)

        self.unet.eval()
        real = self._real_latents(n_samples)
        shape = real.shape[1:]
        gen_chunks = []
        for start in range(0, n_samples, batch):
            b = min(batch, n_samples - start)
            g = torch.Generator().manual_seed(DIST_SEED + start)
            noise = torch.randn((b, *shape), generator=g).to(self.device)
            x = self._denoise_from(
                z=torch.zeros_like(noise),                       # tau0=1 -> z drops out entirely
                noise=noise,
                tau0=1.0,
                spacing=self.spacing_tensor[None].repeat(b, 1),
                modality=torch.full((b,), self.modality_code, dtype=torch.long, device=self.device),
                n_steps=n_steps,
                cfg_scale=cfg_scale,
            )
            gen_chunks.append(x.float().cpu())
        gen = torch.cat(gen_chunks, dim=0)

        fd = float(FIDMetric()(self._pooled_features(gen), self._pooled_features(real)))
        w1 = self._channel_w1(gen, real)
        summary = {
            "epoch": epoch,
            "n_samples": n_samples,
            "num_inference_steps": n_steps,
            "cfg_guidance_scale": cfg_scale,
            "latent_frechet_distance": fd,
            "latent_channel_w1": w1,
            "gen_mean_per_channel": gen.mean(dim=(0, 2, 3, 4)).tolist(),
            "gen_std_per_channel": gen.std(dim=(0, 2, 3, 4)).tolist(),
            "real_mean_per_channel": real.mean(dim=(0, 2, 3, 4)).tolist(),
            "real_std_per_channel": real.std(dim=(0, 2, 3, 4)).tolist(),
        }
        with open(os.path.join(self.img_dir, f"dist_epoch_{epoch:04d}.json"), "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(
            f"[dist] n={n_samples} | latent Frechet={fd:.4f} | channel W1={w1:.4f} | "
            f"gen std={[round(v, 3) for v in summary['gen_std_per_channel']]} "
            f"real std={[round(v, 3) for v in summary['real_std_per_channel']]}"
        )
        return {"fd": fd, "w1": w1}

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

            i2i, dst = None, None
            if (epoch % self.img2img_interval == 0) or (epoch == self.n_epochs - 1):
                i2i = self.img2img(epoch)
            if (epoch % self.dist_interval == 0) or (epoch == self.n_epochs - 1):
                dst = self.dist_check(epoch)
            i2i = i2i or {"lat_l2_rel": -1.0, "mse": -1.0, "psnr": -1.0, "ssim": -1.0}
            dst = dst or {"fd": -1.0, "w1": -1.0}

            if self.is_main:
                lr = self.optimizer.param_groups[0]["lr"]
                append_csv(
                    self.train_log_csv,
                    [epoch, lr]
                    + [tr[k] for k in LOSS_KEYS]
                    + [te[k] for k in TEST_LOSS_KEYS]
                    + [i2i["lat_l2_rel"], i2i["mse"], i2i["psnr"], i2i["ssim"]]
                    + [dst["fd"], dst["w1"], self.n_train, self.n_test],
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
