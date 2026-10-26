# Datasets

The ScarceVid benchmark consists of six existing datasets. Download them from their original sources (see the
references in the paper) and agree to their licenses; we do not redistribute any data.

| Dataset | Task | Classes | `--dataset_name` |
|---|---|---|---|
| CAER | emotion classification | 7 | `CAER` |
| NurViD | nursing procedure classification | 51 | `NurVid` |
| IndustReal | assembly step recognition | 75 | `IndustReal` |
| MammAlps | wildlife behavior recognition | 11 | `MammAlps` |
| ScanNet | depth estimation | – | `scannetv2` |
| VSPW | semantic segmentation | 124 | `vspw_v2` |
| Waymo Open | box tracking | – | `waymo_open_iv_v2` |
| CholecTrack20 | box tracking | – | `cholectrack20_iv` |

Any other dataset can be used by converting it to one of the formats below; `--dataset_name` is the name of its
folder in `--data_dir`. [Prepared datasets](#prepared-datasets) lists the exact contents of our prepared
version of each dataset.

## Video format

All videos are re-encoded to **DNxHR HQ** in a `.mov` container, with the short side scaled to 256 px and both
sides rounded to a multiple of 16 (yuv422p, no audio). DNxHR is an intra-frame codec: every frame is a keyframe,
so the frames sampled for a clip can be decoded directly, without decoding from the previous keyframe as for
H.264/H.265. This makes random frame access with [TorchCodec](https://github.com/pytorch/torchcodec) fast, at the
cost of larger files.

`scripts/utils/recodec.sh` re-encodes all `.mp4` files in `INPUT_DIR` to `OUTPUT_DIR` (set both at the top of
the script) and keeps the frame rate of the source video. `scripts/utils/recodec_nurvid.sh` additionally
resamples to a constant 25 fps, which keeps time-based segment annotations exact for videos with a variable
frame rate. The box tracking converters crop the frames to 256x256 and write them at 5 fps (see below).

## Annotation formats

`--data_dir` points to a folder with one subfolder per dataset. Every dataset folder contains the videos in
`videos/` and one annotation file per split (`train`, `val`, `test`); the format depends on the task. Video
references are file names in `videos/` without the `.mov` extension.

### Classification

```
CAER/
├── videos/<vid_name>.mov
└── train.json, val.json, test.json
```

Each split is a list with one entry per sample, with the class index (`0` to `num_classes - 1`) as label:

```json
[
  {"vid_name": "0000001", "label": 3},
  {"vid_name": "xyz", "label": 12, "start_time": 4.5, "end_time": 19.0},
  {"vid_name": "abc", "label": 40, "start_frame": 120, "end_frame": 480}
]
```

When a sample is a segment of a longer video, add either `start_time`/`end_time` (seconds) or
`start_frame`/`end_frame` (inclusive); frames are then only sampled within that segment. Use one of the two
for all entries of a file. Set `--num_classes` to the number of classes.

### Depth estimation

```
scannetv2/
├── videos/<scene>.mov
├── labels/<scene>.h5
└── train.txt, val.txt, test.txt
```

Each split is a text file with one video name per line. `labels/<scene>.h5` contains a dataset `depth` of shape
`(num_frames, H, W)` with the depth of every frame in millimeters as `uint16`, at the same resolution and number
of frames as the video. Depth is converted to meters when loading. Pixels with a depth outside
(`--min_depth`, `--max_depth`), default 0.001 to 10 m, are ignored in the loss and the metrics, so store missing
depth as 0.

### Semantic segmentation

```
vspw_v2/
├── videos/<clip>.mov
├── labels/<clip>.h5
└── train.txt, val.txt, test.txt
```

The same layout as depth estimation, but `labels/<clip>.h5` contains a dataset `mask` of shape
`(num_frames, H, W)` with the class index of every pixel as `uint8` (`0` to `num_classes - 1`), and
`--seg_void_label` (default 255) for pixels that are ignored in the loss and the metric. Set `--num_classes` to the
number of classes.

### Box tracking

```
waymo_open/
├── videos/<clip>.mov
└── train.json, val.json, test.json
```

Each split maps a video file name to its tracks:

```json
{
  "clip_0001.mov": {
    "img_size": [256, 256],
    "query_frames": [0, 5, 10],
    "tracks": {
      "7":  {"frames": [0, 1, 2, 3], "boxes": [[12.0, 30.5, 80.2, 96.0], [13.1, 31.0, 81.0, 97.2], ...]},
      "12": {"frames": [5, 6, 7],    "boxes": [[...], ...]}
    }
  }
}
```

- `img_size` is the `[width, height]` of the stored video. Box tracking uses no augmentation: frames are resized
  and center-cropped to `--input_size`, and boxes are scaled by `input_size / width`. Store the videos square
  (crop them before encoding) so that the boxes stay aligned.
- `tracks` contains, per object, the frames (0-based, increasing) in which it is visible and its box in each of
  these frames as `[x_min, y_min, x_max, y_max]` in pixels of the stored video.
- `query_frames` are the frames at which a clip of `--clip_len` frames can start. The objects visible at the query
  frame are tracked through the clip: at most `--max_queries` objects, largest first, that are at least
  `--box_area_frac` of the frame at the query frame and visible in at least `--box_min_visible_frac` of the clip.
  The model receives their boxes at the query frame and predicts their boxes in the following frames.
- Training uses every query frame, validation the middle query frame of every video, and testing
  non-overlapping clips over the query frames.

`scripts/utils/convert_iv.sh` (Waymo Open) and `scripts/utils/convert_cholec_iv.sh` (CholecTrack20) create
datasets in this format. Their files contain three more fields that the data loader does not use: `fps` and
`num_frames` per video, and `type` (object class) per track.

## Prepared datasets

We checked our prepared version of every dataset with `ffprobe` (all videos) and by reading all annotation files.
Your version should match the tables below.

All videos in all datasets are DNxHR HQ (`dnxhd` codec, profile `DNXHR HQ`), `yuv422p`, in a `.mov` container,
with a single video stream and no audio. The short side is 256 px and both sides are a multiple of 16. The
`nb_frames` header matches the number of decoded frames, and `ffprobe` reports no errors.

### Videos

| Dataset | Videos | Resolution (W x H) | Frame rate | Frames per video (min / median / max) |
|---|---|---|---|---|
| CAER | 13,175 | 448x256 (6,848), 336x256 (4,305), 352x256 (1,388), 384x256 (634) | source frame rate: 23.976 (6,215), 24 (3,319), 30 (2,421), 25 (685), 29.97 (535) | 36 / 68 / 753 |
| NurViD | 5,610 | 26 sizes, mostly 448x256 (4,428), 256x448 (422), 336x256 (338) | constant 25 | 1 / 405 / ~19,400 |
| IndustReal | 86 | 448x256 | 10 | 1,129 / 2,364 / 5,276 |
| MammAlps | 6,135 | 256x256 | 30 | 1 / 85 / 2,898 |
| ScanNet | 1,613 | 336x256 | 30 | 115 / 1,394 / 8,336 |
| VSPW | 3,536 | 12 sizes, mostly 448x256 (3,449) | 30 | 11 / 45 / 241 |
| Waymo Open | 1,000 | 256x256 | 5 | 86 / 99 / 100 |
| CholecTrack20 | 869 | 256x256 | 5 (content sampled at 1 fps) | 16 / 43 / 1,003 |

A few NurViD and MammAlps videos are shorter than a clip, down to a single frame. Frames are then sampled
repeatedly.

### Annotations

| Dataset | Files | Train / val / test | Notes |
|---|---|---|---|
| CAER | `{train,val,test}.json`: `vid_name`, `label` | 9,222 / 1,316 / 2,637 videos | labels 0-6, all 7 classes in every split; every video is used once |
| NurViD | `{train,val,test}.json`: `vid_name`, `label` | 3,899 / 586 / 1,121 videos | labels 0-50, all 51 classes in every split; 4 videos in `videos/` are not in any split |
| IndustReal | `{train,val,test}.json`: `vid_name`, `label`, `start_frame`, `end_frame` | 3,667 / 1,928 / 3,678 segments from 36 / 16 / 32 videos | labels 0-74 (`--num_classes 75`), with 72 / 65 / 68 classes present; segments are 2-254 frames (median 15-18); 2 videos are not in any split; one train segment ends after the last frame of its video and is clipped when loading |
| MammAlps | `{train,val,test}.json`: `vid_name`, `label` | 4,205 / 686 / 1,244 videos | labels 0-10, with 11 / 9 / 11 classes present; every video is used once |
| ScanNet | `{train,val,test}.txt`, `labels/<scene>.h5` (`depth`, `uint16`, mm) | 1,201 / 312 / 100 scenes | no overlap between splits; every label has the same number of frames and resolution as its video; maximum depth 9.996 m; a median of 9.5% of the pixels per scene (1.1-41.9%) have no depth (0) |
| VSPW | `{train,val,test}.txt`, `labels/<clip>.h5` (`mask`, `uint8`) | 2,806 / 343 / 343 clips | `test.txt` is identical to `val.txt`: VSPW does not release test labels, so we report results on the validation set. The 387 test videos are in `videos/` without labels and are not used. Mask values are 0-123 and 255 (void). The labels have the same resolution as the video; 2 label files have one frame more than their video, which is ignored. |
| Waymo Open | `{train,val,test}.json` | 676 / 172 / 172 videos | `test.json` is identical to `val.json`; `img_size` is `[256, 256]` for all videos; 1-64 tracks per video (median 15); every query frame has at least 16 frames left; all boxes lie inside the frame and have positive size |
| CholecTrack20 | `{train,val,test}.json`, one video per instrument track (`VID<nn>_track<k>.mov`) | 361 / 103 / 405 videos | 1 track per video; `img_size` is `[256, 256]` for all videos; every query frame has at least 16 frames left; all boxes lie inside the frame and have positive size; `fps` in the json is 1, the sampling rate of the frames, while the video is stored at 5 fps |

## Recommendation for SLURM clusters

Reading thousands of video and label files from a shared network file system can be much slower than training
itself and puts a high load on the file system. We recommend storing every dataset as a single tar file and
extracting it to the node-local disk of the GPU node at the start of each job:

```bash
# once, on the shared file system
tar -cf /path/to/datasets/CAER.tar -C /path/to/datasets/CAER .

# at the start of the job script, before main.py
mkdir -p $TMPDIR/data/CAER
tar -xf /path/to/datasets/CAER.tar -C $TMPDIR/data/CAER
DATA_DIR=$TMPDIR/data
```

`$TMPDIR` is the node-local scratch directory on most clusters; check the documentation of your cluster for its
name and size.

## Checking the data

The data test loads a few batches from the train, val and test loaders and saves the frames (and boxes for box
tracking) as images to `--output_dir`:

```bash
python main.py --state_type test_data --config config/classification.yaml \
    --data_dir ./data --dataset_name CAER --num_classes 7 --output_dir ./output/data_test
```

`scripts/utils/run_data_test.sh` runs the same test as a SLURM job.
