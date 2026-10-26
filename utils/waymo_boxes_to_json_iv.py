"""
Output (one JSON per split; keys are the video filenames in the shared
`videos/` folder):

    {
      "<context>.mov": {
        "num_frames": <T>,
        "img_size": [<crop_size>, <crop_size>],
        "fps": <target_fps>,
        "query_frames": [f, ...],            # valid clip first-frames
        "tracks": {
          "<camera_object_id>": {
            "type": <int>,                       # Waymo camera box type
            "frames": [f0, f1, ...],             # frame indices (into the .mov)
            "boxes":  [[x1, y1, x2, y2], ...]    # aligned with `frames`, in
                                                 # crop_size x crop_size pixels,
                                                 # [xmin, ymin, xmax, ymax]
          },
          ...
        }
      },
      ...
    }
"""

import argparse
import json
import math
from pathlib import Path

import pyarrow.parquet as pq

CAMERA_NAMES = {1: "FRONT", 2: "FRONT_LEFT", 3: "FRONT_RIGHT", 4: "SIDE_LEFT", 5: "SIDE_RIGHT"}

NATIVE_FPS = 10.0  
AREA_FRAC = 0.005
WINDOW = 16          
IN_WINDOW_FRAC = 0.7 
MAX_GAP = 5          
COOCCUR_MIN = 10     
MAX_OBJECTS = 25     
DEFAULT_ORIG_W = 1920
DEFAULT_ORIG_H = 1280

def find_column_name(all_columns, suffix):
    matches = [c for c in all_columns if c.endswith(suffix)]
    if not matches:
        raise KeyError(
            f"No column ending with '{suffix}' found. Available: {list(all_columns)}"
        )
    return matches[0]


def kept_timestamps(image_parquet, camera_id, target_fps):
    schema_names = pq.ParquetFile(image_parquet).schema_arrow.names
    cam_col = find_column_name(schema_names, "key.camera_name")
    ts_col = find_column_name(schema_names, "key.frame_timestamp_micros")

    table = pq.read_table(image_parquet, columns=[cam_col, ts_col]).to_pandas()
    ts = sorted(table.loc[table[cam_col] == camera_id, ts_col].tolist())
    if not ts:
        return []
    step = max(1, round(NATIVE_FPS / target_fps))
    return ts[::step]


def transform_box(cx, cy, w, h, sx, sy, left, top, crop_size):
    """Map an original-pixel center box through resize + center-crop, clip to the
    crop, and return [x1, y1, x2, y2] plus the clipped area (in crop pixels).

    sx, sy: resize scale factors (resize_w/orig_w, resize_h/orig_h).
    left, top: center-crop offsets in resized pixels.
    """
    # center box -> corners in original pixels
    x1 = (cx - w / 2.0) * sx - left
    y1 = (cy - h / 2.0) * sy - top
    x2 = (cx + w / 2.0) * sx - left
    y2 = (cy + h / 2.0) * sy - top

    # clip to the crop
    cx1 = min(max(x1, 0.0), crop_size)
    cy1 = min(max(y1, 0.0), crop_size)
    cx2 = min(max(x2, 0.0), crop_size)
    cy2 = min(max(y2, 0.0), crop_size)

    area = max(0.0, cx2 - cx1) * max(0.0, cy2 - cy1)
    return [cx1, cy1, cx2, cy2], area


def longest_gap(present, q, window):
    """Longest run of consecutive frames in [q, q+window-1] absent from `present`.
    Absences at either end of the window count: the object is expected to be
    trackable across the whole window, so a trailing disappearance is a gap."""
    longest = run = 0
    for j in range(q, q + window):
        if j in present:
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    return longest


def select_query_frames(tracks, n, window, per_object_min, max_gap, cooccur_min, max_objects):
    """q is listed iff every one of the objects is present for
    >= per_object_min frames with no gap longer than max_gap, and they all co-occur
    (are simultaneously present) in >= cooccur_min frames."""
    present = {oid: set(t["boxes"]) for oid, t in tracks.items()}
    query_frames = []
    for q in range(0, n - window + 1):
        window_range = range(q, q + window)
        S = [oid for oid in present if q in present[oid]]
        if not S:
            continue
        if len(S) > max_objects:  
            S.sort(key=lambda oid: tracks[oid]["areas"][q], reverse=True)
            S = S[:max_objects]

        if any(sum(1 for j in window_range if j in present[oid]) < per_object_min
               or longest_gap(present[oid], q, window) > max_gap
               for oid in S):
            continue

        if sum(1 for j in window_range if all(j in present[oid] for oid in S)) < cooccur_min:
            continue

        query_frames.append(q)
    return query_frames


def process_segment(box_parquet, image_parquet, camera_id, target_fps,
                    sx, sy, left, top, crop_size, per_object_min, window, max_gap,
                    cooccur_min, max_objects):
    """Return (num_frames, tracks_dict, query_frames) for one segment, or
    (0, {}, []) if the video has no frames, no box survives the 0.5% filter, or
    no frame qualifies as a query frame (in which case the segment is excluded)."""
    kept = kept_timestamps(image_parquet, camera_id, target_fps)
    if not kept:
        return 0, {}, []
    ts_to_idx = {ts: i for i, ts in enumerate(kept)}
    n = len(kept)

    names = pq.ParquetFile(box_parquet).schema_arrow.names
    cam_col = find_column_name(names, "key.camera_name")
    ts_col = find_column_name(names, "key.frame_timestamp_micros")
    oid_col = find_column_name(names, "key.camera_object_id")
    cx_col = find_column_name(names, "box.center.x")
    cy_col = find_column_name(names, "box.center.y")
    sxc_col = find_column_name(names, "box.size.x")
    syc_col = find_column_name(names, "box.size.y")
    type_col = find_column_name(names, "].type")

    # rename to safe identifiers
    rename = {cam_col: "cam", ts_col: "ts", oid_col: "oid", cx_col: "cx",
              cy_col: "cy", sxc_col: "sw", syc_col: "sh", type_col: "typ"}
    df = pq.read_table(box_parquet, columns=list(rename)).to_pandas().rename(columns=rename)
    df = df[df["cam"] == camera_id]

    min_area = AREA_FRAC * crop_size * crop_size
    tracks = {}
    for row in df.itertuples(index=False):
        idx = ts_to_idx.get(row.ts)
        if idx is None:  # box on a frame the video dropped during downsampling
            continue
        box, area = transform_box(
            row.cx, row.cy, row.sw, row.sh, sx, sy, left, top, crop_size
        )
        if area < min_area:  # global 0.5% filter: too small (or outside the crop)
            continue
        t = tracks.get(row.oid)
        if t is None:
            t = {"type": int(row.typ) if row.typ == row.typ else None,
                 "boxes": {}, "areas": {}}
            tracks[row.oid] = t
        # If a track somehow appears twice on one frame, keep the larger box.
        if idx not in t["areas"] or area > t["areas"][idx]:
            t["boxes"][idx] = [round(v, 2) for v in box]
            t["areas"][idx] = area

    if not tracks:
        return 0, {}, []

    query_frames = select_query_frames(
        tracks, n, window, per_object_min, max_gap, cooccur_min, max_objects
    )
    if not query_frames:
        return 0, {}, []

    # Flatten to frame-ordered lists (parquet row order is not guaranteed).
    out_tracks = {}
    for oid, t in tracks.items():
        idxs = sorted(t["boxes"])
        out_tracks[oid] = {"type": t["type"],
                           "frames": idxs,
                           "boxes": [t["boxes"][i] for i in idxs]}

    return n, out_tracks, query_frames


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--box-dir", type=Path, required=True,
                    help="Dir of camera_box parquet files, e.g. .../training/camera_box")
    ap.add_argument("--image-dir", type=Path, required=True,
                    help="Matching camera_image dir (for frame timestamps), "
                         "e.g. .../training/camera_image")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output JSON path, e.g. .../train_iv.json")
    ap.add_argument("--camera", type=int, default=1, help="Camera id (default 1=FRONT)")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--resize-h", type=int, default=256)
    ap.add_argument("--resize-w", type=int, default=384)
    ap.add_argument("--crop-size", type=int, default=256)
    ap.add_argument("--orig-w", type=int, default=DEFAULT_ORIG_W)
    ap.add_argument("--orig-h", type=int, default=DEFAULT_ORIG_H)
    ap.add_argument("--video-suffix", type=str, default=".mov",
                    help="Suffix appended to the context name to form the JSON key")
    ap.add_argument("--window", type=int, default=WINDOW,
                    help="Tracking window length the dataloader samples")
    ap.add_argument("--in-window-frac", type=float, default=IN_WINDOW_FRAC,
                    help="Fraction of the window each first-frame object must appear in; "
                         "per-object threshold = ceil(frac*window)")
    ap.add_argument("--max-gap", type=int, default=MAX_GAP,
                    help="Longest continuous missing run allowed per first-frame object")
    ap.add_argument("--cooccur", type=int, default=COOCCUR_MIN,
                    help="Frames in which all first-frame objects must co-occur")
    ap.add_argument("--max-objects", type=int, default=MAX_OBJECTS,
                    help="First-frame objects used per window (largest by area)")
    ap.add_argument("--start", type=int, default=0,
                    help="Start index into the sorted box-file list (for array jobs)")
    ap.add_argument("--end", type=int, default=None,
                    help="End index into the sorted box-file list (for array jobs)")
    args = ap.parse_args()

    if args.output.exists():
        raise SystemExit(
            f"{args.output} already exists -- refusing to overwrite. "
            f"Move it aside or pass a different --output."
        )

    sx = args.resize_w / args.orig_w
    sy = args.resize_h / args.orig_h
    left = (args.resize_w - args.crop_size) // 2
    top = (args.resize_h - args.crop_size) // 2

    per_object_min = math.ceil(args.in_window_frac * args.window)

    box_files = sorted(args.box_dir.glob("*.parquet"))[args.start:args.end]
    print(f"Processing {len(box_files)} segment(s) from {args.box_dir}", flush=True)
    print(f"Rule (InternVideo-Next): window={args.window}, each first-frame object "
          f">= {per_object_min} frames (ceil({args.in_window_frac}*{args.window})), "
          f"gap <= {args.max_gap}, all co-occur >= {args.cooccur}, "
          f"<= {args.max_objects} objects/window, area >= {AREA_FRAC:.1%} of crop", flush=True)

    out = {}
    for i, bf in enumerate(box_files):
        context = bf.stem
        image_parquet = args.image_dir / bf.name
        key = f"{context}{args.video_suffix}"
        try:
            if not image_parquet.exists():
                print(f"[skip] no camera_image parquet for {context}", flush=True)
                continue
            num_frames, tracks, query_frames = process_segment(
                bf, image_parquet, args.camera, args.fps,
                sx, sy, left, top, args.crop_size,
                per_object_min, args.window, args.max_gap,
                args.cooccur, args.max_objects,
            )
            if num_frames == 0 or not tracks:
                print(f"[skip] no qualifying query frames for {context}", flush=True)
                continue
            out[key] = {
                "num_frames": num_frames,
                "img_size": [args.crop_size, args.crop_size],
                "fps": args.fps,
                "query_frames": query_frames,
                "tracks": tracks,
            }
            print(f"[{i + 1}/{len(box_files)}] {key}: {num_frames} frames, "
                  f"{len(tracks)} tracks, {len(query_frames)} query frames", flush=True)
        except Exception as e:
            print(f"[error] {context}: {e}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f)
    print(f"Wrote {len(out)} segments to {args.output}", flush=True)


if __name__ == "__main__":
    main()
