"""
Convert CholecTrack20 into single-object tracking clips + JSON, in the SAME
format as the Waymo pipeline, and following the InternVideo-Next box rules.

Output (one JSON per split; keys are the clip video filenames in the shared
`videos/` folder), identical schema to the Waymo JSON:

    {
      "<VID>_track<t>.mov": {
        "num_frames": <T>,
        "img_size": [<crop_size>, <crop_size>],
        "fps": 1,
        "query_frames": [f, ...],
        "tracks": {
          "<t>": {
            "type": <instrument id>,
            "frames": [f0, f1, ...],
            "boxes":  [[x1, y1, x2, y2], ...]   # crop_size px, [xmin,ymin,xmax,ymax]
          }
        }
      },
      ...
    }
"""

import argparse
import io
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ORIG_W, ORIG_H = 854, 480          # every CholecTrack20 video is 854x480
TRACK_KEY = "visibility_track"      # perspective used to define tracks
WINDOW = 16                         # clip / tracking window length (frames)
IN_WINDOW_FRAC = 0.7                # each track must be present for >= this
                                    # fraction of the window: ceil(0.7*16)=12
MAX_GAP = 5                         # longest run of consecutive missing frames
                                    # allowed within the window
AREA_FRAC = 0.005                   # global filter: boxes below 0.5% of the crop
                                    # are dropped on every frame


def resize_and_crop(pil_img, target_h, target_w, crop_size):
    """Resize a PIL image to (target_w, target_h) then center-crop a square."""
    img = pil_img.convert("RGB").resize((target_w, target_h), Image.LANCZOS)
    left = (target_w - crop_size) // 2
    top = (target_h - crop_size) // 2
    img = img.crop((left, top, left + crop_size, top + crop_size))
    return np.asarray(img, dtype=np.uint8)  # HxWx3, RGB


def transform_box_tlwh(xn, yn, wn, hn, sx, sy, left, top, crop_size):
    """Map a normalized tlwh box through resize + center-crop, clip to the crop,
    and return ([x1, y1, x2, y2], clipped_area) in crop pixels."""
    x1 = (xn * ORIG_W) * sx - left
    y1 = (yn * ORIG_H) * sy - top
    x2 = ((xn + wn) * ORIG_W) * sx - left
    y2 = ((yn + hn) * ORIG_H) * sy - top

    cx1 = min(max(x1, 0.0), crop_size)
    cy1 = min(max(y1, 0.0), crop_size)
    cx2 = min(max(x2, 0.0), crop_size)
    cy2 = min(max(y2, 0.0), crop_size)

    area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return [cx1, cy1, cx2, cy2], area


def write_video_ffmpeg(frames, out_path, fps, size):
    """Pipe raw RGB frames into ffmpeg, encoding to DNxHR MOV (same as Waymo)."""
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

    tmp_path.rename(out_path)


def load_frames_from_pngs(frames_dir, framenums, resize_hw, crop_size):
    """Read the requested frame numbers as {framenum: cropped RGB np} from the
    per-video Frames/ folder (files named `<framenum:06d>.png`)."""
    target_h, target_w = resize_hw
    cache = {}
    for n in framenums:
        p = frames_dir / f"{n:06d}.png"
        with Image.open(p) as img:
            cache[n] = resize_and_crop(img, target_h, target_w, crop_size)
    return cache


def load_frames_from_mp4(mp4_path, framenums, resize_hw, crop_size):
    """Decode the requested (0-based) frame indices from a long mp4 into
    {framenum: cropped RGB np}. Sequential decode (robust to codec seeking),
    stopping once the last needed frame is read."""
    import cv2  # local import so PNG-only runs don't require OpenCV

    target_h, target_w = resize_hw
    needed = set(framenums)
    last = max(needed)
    cache = {}

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {mp4_path}")
    idx = 0
    try:
        while idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in needed:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cache[idx] = resize_and_crop(Image.fromarray(rgb), target_h, target_w, crop_size)
            idx += 1
    finally:
        cap.release()

    missing = needed - cache.keys()
    if missing:
        raise RuntimeError(f"{mp4_path}: {len(missing)} needed frames not decoded "
                           f"(e.g. {sorted(missing)[:5]}); decoded up to idx {idx - 1}")
    return cache


def extract_tracks(ann, sx, sy, left, top, crop_size):
    """From a CholecTrack20 `annotations` dict, build per-track occurrence data
    on the sampled (1 fps) frame sequence, keeping only boxes >= 0.5% of the crop.

    Returns (seq, tracks) where:
      seq    : sorted list of original frame numbers; index i == frame i.
      tracks : {track_id: {"type": instrument_id,
                           "boxes": {i: [x1,y1,x2,y2]},
                           "areas": {i: clipped_area}}}
    """
    seq = sorted(int(k) for k in ann)
    idx_of = {n: i for i, n in enumerate(seq)}
    min_area = AREA_FRAC * crop_size * crop_size

    tracks = {}
    for k, dets in ann.items():
        i = idx_of[int(k)]
        for d in dets:
            t = d.get(TRACK_KEY)
            if t is None or t < 0:
                continue
            box, area = transform_box_tlwh(*d["tool_bbox"], sx, sy, left, top, crop_size)
            if area < min_area:  # global 0.5% filter (too small or outside the crop)
                continue
            tr = tracks.get(t)
            if tr is None:
                tr = {"type": int(d.get("instrument", -1)), "boxes": {}, "areas": {}}
                tracks[t] = tr
            # If the same track appears twice in one frame, keep the larger box.
            if i not in tr["areas"] or area > tr["areas"][i]:
                tr["boxes"][i] = [round(v, 2) for v in box]
                tr["areas"][i] = area
    return seq, tracks


def longest_gap(present, q, window):
    """Longest run of consecutive frames in [q, q+window-1] absent from `present`.
    Absences at either end count: the track is expected across the whole window."""
    longest = run = 0
    for j in range(q, q + window):
        if j in present:
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    return longest


def build_samples(seq, tracks, crop_size, video_stem, video_suffix,
                  min_in_window, window, max_gap):
    """Turn per-track occurrences into clip samples.

    Every stored box is already >=0.5%, so the query candidates are just the
    frames where the track is present. A query q is valid if the full window
    [q, q+window-1] exists, >= min_in_window of its frames have the track, and no
    missing run within it is longer than max_gap.

    Returns list of (clip_key, entry_dict, clip_framenums) where clip_framenums
    is the list of ORIGINAL frame numbers that make up the clip video, in order.
    """
    n = len(seq)
    samples = []

    for t, tr in sorted(tracks.items()):
        present = set(tr["areas"])
        query_cands = sorted(present)

        valid_q = []
        for q in query_cands:
            if q + window - 1 > n - 1:      # not enough following frames for a full window
                continue
            hit = sum(1 for j in range(q, q + window) if j in present)
            if hit < min_in_window:
                continue
            if longest_gap(present, q, window) > max_gap:
                continue
            valid_q.append(q)
        if not valid_q:
            continue

        start = min(valid_q)
        end = max(valid_q) + window - 1     # inclusive; guaranteed <= n-1
        clip_indices = range(start, end + 1)
        clip_framenums = [seq[i] for i in clip_indices]

        frames, boxes = [], []
        for i in clip_indices:
            if i in tr["boxes"]:
                frames.append(i - start)
                boxes.append(tr["boxes"][i])

        entry = {
            "num_frames": len(clip_framenums),
            "img_size": [crop_size, crop_size],
            "fps": 1,
            "query_frames": [q - start for q in valid_q],
            "tracks": {str(t): {"type": tr["type"], "frames": frames, "boxes": boxes}},
        }
        clip_key = f"{video_stem}_track{t}{video_suffix}"
        samples.append((clip_key, entry, clip_framenums))

    return samples


def process_video(video_dir, split, resize_hw, crop_size, sx, sy, left, top,
                  video_suffix, min_in_window, window, max_gap, out_video_dir, encode_fps):
    """Process one VID* folder: build its clip samples, write the clip videos,
    and return the JSON entries {clip_key: entry}."""
    stem = video_dir.name
    json_path = video_dir / f"{stem}.json"
    ann = json.load(open(json_path))["annotations"]

    seq, tracks = extract_tracks(ann, sx, sy, left, top, crop_size)
    samples = build_samples(seq, tracks, crop_size, stem, video_suffix,
                            min_in_window, window, max_gap)
    if not samples:
        print(f"[skip] {stem}: no qualifying tracks", flush=True)
        return {}

    # Decode/read every frame number any clip of this video needs, exactly once.
    needed = sorted({fn for _, _, fns in samples for fn in fns})
    if split == "Testing":
        frame_cache = load_frames_from_mp4(video_dir / f"{stem}.mp4", needed,
                                           resize_hw, crop_size)
    else:
        frame_cache = load_frames_from_pngs(video_dir / "Frames", needed,
                                            resize_hw, crop_size)

    entries = {}
    for clip_key, entry, clip_framenums in samples:
        out_path = out_video_dir / clip_key
        if not out_path.exists():
            frames = (frame_cache[fn] for fn in clip_framenums)
            write_video_ffmpeg(frames, out_path, encode_fps, (crop_size, crop_size))
        entries[clip_key] = entry

    print(f"[done] {stem}: {len(samples)} clips from {len(tracks)} tracks", flush=True)
    return entries


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--split-dir", type=Path, required=True,
                    help="A CholecTrack20 split dir, e.g. .../cholectrack20/Training")
    ap.add_argument("--split", choices=["Training", "Validation", "Testing"],
                    default=None, help="Frame source: Testing decodes mp4, else PNGs. "
                                       "Defaults to the split dir's name.")
    ap.add_argument("--video-out", type=Path, required=True,
                    help="Output dir for the clip videos, e.g. .../cholectrack20_iv/videos")
    ap.add_argument("--json-out", type=Path, required=True,
                    help="Output JSON path, e.g. .../cholectrack20_iv/train.json")
    ap.add_argument("--resize-h", type=int, default=256)
    ap.add_argument("--resize-w", type=int, default=455,
                    help="round(854*256/480)=455: short side 256, aspect kept")
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--encode-fps", type=int, default=5,
                    help="ffmpeg output framerate (content unaffected; DNxHR "
                         "dislikes 1fps). Semantic fps is stored as 1 in the JSON.")
    ap.add_argument("--in-window-frac", type=float, default=IN_WINDOW_FRAC,
                    help="Fraction of the window the track must appear in; "
                         "threshold = ceil(frac*window)")
    ap.add_argument("--max-gap", type=int, default=MAX_GAP,
                    help="Longest continuous missing run allowed within the window")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--video-suffix", type=str, default=".mov")
    args = ap.parse_args()

    split = args.split or args.split_dir.name
    sx = args.resize_w / ORIG_W
    sy = args.resize_h / ORIG_H
    left = (args.resize_w - args.crop_size) // 2
    top = (args.resize_h - args.crop_size) // 2

    min_in_window = math.ceil(args.in_window_frac * args.window)

    video_dirs = sorted(p for p in args.split_dir.iterdir()
                        if p.is_dir() and p.name.startswith("VID"))
    print(f"Processing {len(video_dirs)} {split} videos from {args.split_dir}", flush=True)
    print(f"Rule (InternVideo-Next): window={args.window}, track present >= {min_in_window} "
          f"frames (ceil({args.in_window_frac}*{args.window})), gap <= {args.max_gap}, "
          f"area >= {AREA_FRAC:.1%} of crop on every frame", flush=True)

    args.video_out.mkdir(parents=True, exist_ok=True)
    out = {}
    for vd in video_dirs:
        try:
            out.update(process_video(
                vd, split, (args.resize_h, args.resize_w), args.crop_size,
                sx, sy, left, top, args.video_suffix,
                min_in_window, args.window, args.max_gap, args.video_out, args.encode_fps,
            ))
        except Exception as e:
            print(f"[error] {vd.name}: {e}", flush=True)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} clips to {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
