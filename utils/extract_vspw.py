import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import numpy as np
from PIL import Image
from tqdm import tqdm


VSPW_ROOT = "/path/to/datasets/vspw/VSPW"       # source dataset
OUT_ROOT  = "/path/to/datasets/vspw_v2"     # destination
MAX_WORKERS = 18
SHORT_SIDE = 256
MULTIPLE   = 16


def compute_target_size(w: int, h: int) -> tuple[int, int]:
    """Fix the short side to SHORT_SIDE, round both dims to nearest MULTIPLE."""
    if w >= h:                       # landscape or square
        new_h = SHORT_SIDE
        new_w = MULTIPLE * round((SHORT_SIDE * w / h) / MULTIPLE)
    else:                            # portrait
        new_w = SHORT_SIDE
        new_h = MULTIPLE * round((SHORT_SIDE * h / w) / MULTIPLE)
    return int(new_w), int(new_h)

def _sorted_frames(directory: Path, ext: str) -> list[Path]:
    """Glob for *.<ext>, skip macOS ._ resource forks, sort numerically."""
    return sorted(
        (p for p in directory.glob(f"*.{ext}") if not p.name.startswith("._")),
        key=lambda p: int(p.stem),
    )


def get_frame_info(frame_dir: Path) -> tuple[int, int, list[Path]]:
    """Return (width, height, sorted_frame_paths) from a directory of jpgs."""
    frames = _sorted_frames(frame_dir, "jpg")
    if not frames:
        raise FileNotFoundError(f"No .jpg frames in {frame_dir}")
    img = cv2.imread(str(frames[0]))
    h, w = img.shape[:2]
    return w, h, frames


def encode_color(frame_dir: Path, out_path: str, W: int, H: int) -> None:
    """Encode origin/*.jpg → .mov (DNxHR‑HQ, yuv422p)."""
    # Build a concat file so frame ordering is deterministic regardless of
    # locale and we handle non‑contiguous numbering gracefully.
    frames = _sorted_frames(frame_dir, "jpg")
    concat_path = out_path + ".concat.txt"
    try:
        with open(concat_path, "w") as fh:
            for f in frames:
                # ffmpeg concat requires single‑quote escaping inside the path
                safe = str(f).replace("'", "'\\''")
                fh.write(f"file '{safe}'\n")

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning",
            "-r", "30",
            "-f", "concat", "-safe", "0", "-i", concat_path,
            "-vf", f"scale={W}:{H}:flags=bilinear,setsar=1,format=yuv422p",
            "-c:v", "dnxhd", "-profile:v", "dnxhr_hq",
            "-fps_mode", "passthrough", "-movflags", "+faststart", "-an",
            "-y", out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(concat_path):
            os.remove(concat_path)


def encode_masks(mask_dir: Path, out_path: str, W: int, H: int) -> None:
    """Encode mask/*.png → .h5  (LZ4, uint8, one frame per row)."""
    mask_files = _sorted_frames(mask_dir, "png")
    if not mask_files:
        return                       # test clips have no masks

    with h5py.File(out_path, "w") as f:
        ds = f.create_dataset(
            "mask",
            shape=(len(mask_files), H, W),
            dtype=np.uint8,
            chunks=(1, H, W),
            **hdf5plugin.LZ4(),
        )
        for i, p in enumerate(mask_files):
            # Use PIL so palette / indexed PNGs are read as class indices
            m = np.asarray(Image.open(p))
            if m.ndim == 3:          # safety: drop channel dim if present
                m = m[:, :, 0]
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            # Remap: 0 ("others") and 255 ("void") → 255 ignore;
            #        valid classes 1‑124 → 0‑123
            m[m == 0] = 255
            m = m - 1            # 1‑124 → 0‑123;  255 → 254
            m[m > 123] = 255    # restore void
            ds[i] = m

def process_clip(clip_name: str) -> str | None:
    clip_dir   = Path(VSPW_ROOT) / "data" / clip_name
    origin_dir = clip_dir / "origin"
    mask_dir   = clip_dir / "mask"

    videos_dir = os.path.join(OUT_ROOT, "videos")
    labels_dir = os.path.join(OUT_ROOT, "labels")

    mov_path = os.path.join(videos_dir, f"{clip_name}.mov")
    h5_path  = os.path.join(labels_dir, f"{clip_name}.h5")

    # skip already‑processed clips (handy for restarts)
    has_mov = os.path.exists(mov_path)
    has_h5  = os.path.exists(h5_path)
    masks_exist = mask_dir.is_dir() and bool(_sorted_frames(mask_dir, "png"))
    if has_mov and (has_h5 or not masks_exist):
        return None

    try:
        orig_w, orig_h, _ = get_frame_info(origin_dir)
        W, H = compute_target_size(orig_w, orig_h)

        if not has_mov:
            encode_color(origin_dir, mov_path, W, H)

        if not has_h5 and masks_exist:
            encode_masks(mask_dir, h5_path, W, H)

        return None
    except Exception as e:
        print(f"FAILED: {clip_name} — {e}")
        return clip_name

if __name__ == "__main__":
    os.makedirs(os.path.join(OUT_ROOT, "videos"), exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "labels"), exist_ok=True)

    data_dir = os.path.join(VSPW_ROOT, "data")
    clips = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, d))
    )
    print(f"Found {len(clips)} clips.  Processing …")

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        results = list(tqdm(pool.map(process_clip, clips), total=len(clips)))

    errors = [r for r in results if r is not None]
    if errors:
        print(f"\nFinished with {len(errors)} errors:")
        for e in errors:
            print(f"  {e}")
    else:
        print("\nAll clips processed successfully.")