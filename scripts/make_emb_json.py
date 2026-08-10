# Write the *_emb.nii.gz.json sidecar files that diff_model_train.py reads.
# diff_model_create_training_data.py does NOT produce these; only the notebook does.
#
#   python scripts/make_emb_json.py --embedding_base_dir /path/to/embeddings

from __future__ import annotations

import argparse
import json
import os

import nibabel as nib

parser = argparse.ArgumentParser(description="Write spacing/modality sidecars for MAISI embeddings")
parser.add_argument("--embedding_base_dir", type=str, required=True)
parser.add_argument("--modality", type=str, default="mri_t2ax", help="Key from configs/modality_mapping.json")
args = parser.parse_args()

count = 0
for root, _, files in os.walk(args.embedding_base_dir):
    for filename in files:
        if not filename.endswith("_emb.nii.gz"):
            continue
        emb_path = os.path.join(root, filename)
        img = nib.load(emb_path)
        data = {
            "dim": [int(v) for v in img.shape[:3]],
            "spacing": [float(v) for v in img.header.get_zooms()[:3]],
            "modality": args.modality,
        }
        with open(emb_path + ".json", "w") as f:
            json.dump(data, f, indent=4)
        count += 1

print(f"wrote {count} sidecar json files under {args.embedding_base_dir}")
