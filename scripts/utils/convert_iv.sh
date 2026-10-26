#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=convert_waymo_iv
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%A.out

# activate your environment here, e.g.: conda activate video-peft

ROOT=/path/to/datasets/waymo_open
SCRIPT=utils/waymo_boxes_to_json_iv.py


python $SCRIPT \
    --box-dir $ROOT/training/camera_box \
    --image-dir $ROOT/training/camera_image \
    --output $ROOT/train_iv.json \
    --camera 1 --fps 5 \
    --resize-h 256 --resize-w 384 --crop-size 256 \
    --window 16 --in-window-frac 0.7 --max-gap 5 --cooccur 10 --max-objects 25

python $SCRIPT \
    --box-dir $ROOT/validation/camera_box \
    --image-dir $ROOT/validation/camera_image \
    --output $ROOT/val_iv.json \
    --camera 1 --fps 5 \
    --resize-h 256 --resize-w 384 --crop-size 256 \
    --window 16 --in-window-frac 0.7 --max-gap 5 --cooccur 10 --max-objects 25
