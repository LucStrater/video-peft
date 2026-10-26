"""
Convert Waymo Open Dataset v2 `camera_image` parquet files into video clips.

For each segment's parquet file, this:
  1. reads frames for the requested camera(s),
  2. downsamples the native ~10 Hz capture to a target fps,
  3. resizes to a fixed height (preserving aspect ratio),
  4. center-crops to a square,
  5. encodes the result to a DNxHR MOV with ffmpeg.
"""

import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

CAMERA_NAMES = {
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
}

NATIVE_FPS = 10.0


def find_column_name(all_columns, suffix):
    matches = [c for c in all_columns if c.endswith(suffix)]
    if not matches:
        raise KeyError(
            f"No column ending with '{suffix}' found. "
            f"Available columns: {list(all_columns)}"
        )
    return matches[0]


def load_segment_frames(parquet_path, camera_id):
    schema_names = pq.ParquetFile(parquet_path).schema_arrow.names

    cam_col = find_column_name(schema_names, "key.camera_name")
    ts_col = find_column_name(schema_names, "key.frame_timestamp_micros")
    img_col = find_column_name(schema_names, "].image")

    table = pq.read_table(parquet_path, columns=[cam_col, ts_col, img_col])
    df = table.to_pandas()

    sub = df[df[cam_col] == camera_id].sort_values(ts_col)
    return list(zip(sub[ts_col].tolist(), sub[img_col].tolist()))


def downsample_to_fps(frames, native_fps, target_fps):
    """Pick evenly-spaced frames to approximate target_fps from native_fps."""
    if not frames:
        return []
    step = max(1, round(native_fps / target_fps))
    return frames[::step]


def resize_and_crop(jpeg_bytes, target_h, target_w, crop_size):
    """Decode JPEG -> resize to (target_w, target_h) -> center-crop square."""
    img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
    img = img.resize((target_w, target_h), Image.LANCZOS)

    left = (target_w - crop_size) // 2
    top = (target_h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))

    return np.asarray(img, dtype=np.uint8)  # HxWx3, RGB


def write_video_ffmpeg(frames, out_path, fps, size):
    """Pipe raw RGB frames into ffmpeg, encoding to DNxHR MOV.

    frames: iterable of HxWx3 uint8 numpy arrays (already cropped to `size`).
    size: (w, h) of each frame, must be square here (w == h == crop_size).
    """
    w, h = size
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{w}x{h}", "-framerate", str(fps),
        "-i", "-",
        "-vf", "format=yuv422p,setsar=1",
        "-c:v", "dnxhd", "-profile:v", "dnxhr_hq",
        "-movflags", "+faststart", "-an",
        "-f", "mov",
        "-y", str(tmp_path),
    ]

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n_written = 0
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
            n_written += 1
    finally:
        proc.stdin.close()
        ret = proc.wait()

    if ret != 0 or n_written == 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed (code {ret}, {n_written} frames) for {out_path}")

    tmp_path.rename(out_path)  # atomic-ish: only appears once fully written


def process_segment(parquet_path, out_dir, cameras, target_fps, resize_hw, crop_size):
    context_name = Path(parquet_path).stem
    target_h, target_w = resize_hw
    single_camera = len(cameras) == 1

    for camera_id in cameras:
        cam_name = CAMERA_NAMES.get(camera_id, str(camera_id))
        if single_camera:
            out_path = out_dir / f"{context_name}.mov"
        else:
            out_path = out_dir / f"{context_name}_{cam_name}.mov"
        if out_path.exists():
            print(f"[skip] already exists: {out_path.name}", flush=True)
            continue

        frames = load_segment_frames(parquet_path, camera_id)
        if not frames:
            print(f"[skip] no frames for camera {cam_name} in {context_name}", flush=True)
            continue

        frames = downsample_to_fps(frames, NATIVE_FPS, target_fps)

        decoded = (
            resize_and_crop(jpeg, target_h, target_w, crop_size)
            for _, jpeg in frames
        )

        write_video_ffmpeg(decoded, out_path, target_fps, (crop_size, crop_size))
        print(f"[done] {out_path.name} ({len(frames)} frames)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, required=True,
                         help="Dir containing camera_image parquet files, "
                              "e.g. .../training/camera_image or .../validation/camera_image")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cameras", type=int, nargs="+", default=[1],
                         help="Camera IDs to extract: 1=FRONT 2=FRONT_LEFT 3=FRONT_RIGHT "
                              "4=SIDE_LEFT 5=SIDE_RIGHT. Default: FRONT only.")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--resize-h", type=int, default=256)
    parser.add_argument("--resize-w", type=int, default=384)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--start", type=int, default=0,
                         help="Start index into sorted file list (for splitting across array jobs)")
    parser.add_argument("--end", type=int, default=None,
                         help="End index into sorted file list (for splitting across array jobs)")
    args = parser.parse_args()

    parquet_files = sorted(args.input_dir.glob("*.parquet"))
    parquet_files = parquet_files[args.start:args.end]
    print(f"Processing {len(parquet_files)} segment(s) from {args.input_dir}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, pf in enumerate(parquet_files):
        print(f"[{i + 1}/{len(parquet_files)}] {pf.name}", flush=True)
        try:
            process_segment(
                pf, args.output_dir, args.cameras, args.fps,
                (args.resize_h, args.resize_w), args.crop_size,
            )
        except Exception as e:
            print(f"[error] {pf.name}: {e}", flush=True)


if __name__ == "__main__":
    main()