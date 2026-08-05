# Build the MAISI dataset.json from a CSV column holding absolute NIfTI paths.
#
#   python scripts/make_datalist.py --csv /path/to/data.csv --out ./dataset_t2.json

from __future__ import annotations

import argparse
import json
import os

import pandas as pd

parser = argparse.ArgumentParser(description="CSV column -> MAISI dataset.json")
parser.add_argument("--csv", type=str, required=True, help="Path to the source CSV")
parser.add_argument("--column", type=str, default="T2WI_AX", help="Column holding the NIfTI paths")
parser.add_argument("--modality", type=str, default="mri_t2", help="Key from configs/modality_mapping.json")
parser.add_argument("--out", type=str, required=True, help="Output dataset json path")
args = parser.parse_args()

series = pd.read_csv(args.csv)[args.column].dropna().astype(str).str.strip()
paths = [p for p in dict.fromkeys(series) if p]

kept = [p for p in paths if os.path.isfile(p)]
missing = [p for p in paths if not os.path.isfile(p)]

with open(args.out, "w") as f:
    json.dump({"training": [{"image": p, "modality": args.modality} for p in kept]}, f, indent=2)

print(f"csv rows: {len(series)}, unique: {len(paths)}, kept: {len(kept)}, missing: {len(missing)}")
for p in missing[:10]:
    print(f"  missing -> {p}")
print(f"wrote {args.out}")
