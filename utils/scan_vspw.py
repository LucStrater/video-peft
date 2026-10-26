"""Patch any odd label values in already-converted h5 files to 255 (ignore)."""

from pathlib import Path
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

LABELS_DIR = Path("/path/to/datasets/vspw_v2/labels")

for path in sorted(LABELS_DIR.glob("*.h5")):
    with h5py.File(path, "r+") as f:
        ds = f["mask"]
        n_fixed_total = 0
        for i in range(ds.shape[0]):
            frame = ds[i]
            bad = (frame > 123) & (frame != 255)
            if bad.any():
                frame[bad] = 255
                ds[i] = frame
                n_fixed_total += int(bad.sum())
        if n_fixed_total:
            print(f"{path.name}: fixed {n_fixed_total} pixels", flush=True)

print("Done.", flush=True)