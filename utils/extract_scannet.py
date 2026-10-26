import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import h5py
import hdf5plugin
import cv2
import numpy as np
from pathlib import Path


W, H = 336, 256


def encode_color(scene_path, out_path):
    color_dir = os.path.join(scene_path, "color")
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-r", "30", "-i", os.path.join(color_dir, "%d.jpg"),
        "-vf", f"scale={W}:{H}:flags=bilinear,setsar=1,format=yuv422p",
        "-c:v", "dnxhd", "-profile:v", "dnxhr_hq",
        "-fps_mode", "passthrough", "-movflags", "+faststart", "-an",
        "-y", out_path
    ]
    subprocess.run(cmd, check=True)


def encode_depth(scene_path, out_path):
    depth_dir = Path(scene_path) / "depth"
    depth_files = sorted(depth_dir.glob("*.png"), key=lambda p: int(p.stem))

    with h5py.File(out_path, "w") as f:
        ds = f.create_dataset(
            "depth", shape=(len(depth_files), H, W),
            dtype=np.uint16, **hdf5plugin.LZ4(),
            chunks=(1, H, W)
        )
        for i, p in enumerate(depth_files):
            d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
            ds[i] = d


def cleanup_folders(scene_path):
    import shutil
    for folder in ["color", "depth"]:
        p = os.path.join(scene_path, folder)
        if os.path.isdir(p):
            shutil.rmtree(p)


def extract_scene(task):
    sens_file, dst_root, reader_path = task
    scene_id = os.path.basename(sens_file).replace('.sens', '')
    tmp_path = os.path.join(dst_root, "tmp", scene_id)
    os.makedirs(tmp_path, exist_ok=True)

    videos_dir = os.path.join(dst_root, "videos")
    labels_dir = os.path.join(dst_root, "labels")
    os.makedirs(videos_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    cmd = [
        'python', reader_path,
        '--filename', sens_file,
        '--output_path', tmp_path,
        '--export_depth_images',
        '--export_color_images'
    ]

    try:
        subprocess.run(cmd, check=True, cwd=os.path.dirname(reader_path))
        encode_color(tmp_path, os.path.join(videos_dir, f"{scene_id}.mov"))
        encode_depth(tmp_path, os.path.join(labels_dir, f"{scene_id}.h5"))
        cleanup_folders(tmp_path)
        os.rmdir(tmp_path)
        return None
    except Exception as e:
        print(f"FAILED: {scene_id} — {e}")
        try:
            cleanup_folders(tmp_path)
        except:
            pass
        return scene_id


if __name__ == "__main__":
    dir_mapping = {
        '/path/to/datasets/ScanNet/scans/': '/path/to/datasets/ScanNet/scans_extracted/'
        # '/path/to/datasets/ScanNet/scans_test/': '/path/to/datasets/ScanNet/scans_test_extracted/'
    }
    reader_path = '/path/to/datasets/ScanNet/SensReader/python/reader.py'

    tasks = []
    for src_dir, dst_root in dir_mapping.items():
        if os.path.exists(src_dir):
            for root, _, files in os.walk(src_dir):
                for file in files:
                    if file.endswith(".sens"):
                        tasks.append((os.path.join(root, file), dst_root, reader_path))

    print(f"Found {len(tasks)} scenes. Extracting, encoding & cleaning up...")
    with ProcessPoolExecutor(max_workers=18) as executor:
        results = list(tqdm(executor.map(extract_scene, tasks), total=len(tasks)))
    errors = [res for res in results if res is not None]
    if errors:
        print(f"\nFinished with {len(errors)} errors. Check SLURM output for details.")
    else:
        print("\nAll extractions complete.")