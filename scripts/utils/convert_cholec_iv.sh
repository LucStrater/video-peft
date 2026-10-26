#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=convert_cholec_iv
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%A.out

# activate your environment here, e.g.: conda activate video-peft

SRC=/path/to/datasets/cholectrack20        # untouched source (PNGs + mp4 + per-video json)
OUT=/path/to/datasets/cholectrack20_iv     # clip videos -> $OUT/videos, annotations -> $OUT/*.json
SCRIPT=utils/cholec_to_clips_iv.py

for split in Training Validation Testing; do
    case $split in
        Training)   name=train ;;
        Validation) name=val ;;
        Testing)    name=test ;;
    esac
    echo "=== $split -> $name.json ==="
    python $SCRIPT \
        --split-dir $SRC/$split \
        --video-out $OUT/videos \
        --json-out $OUT/$name.json \
        --in-window-frac 0.7 --max-gap 5 --window 16
done
echo "Done."
