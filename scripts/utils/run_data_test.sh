#!/bin/bash
# Load a few batches from the train, val and test loaders of a dataset and save the frames (and boxes for box
# tracking) as images to OUTPUT_DIR (see docs/DATASET.md: Checking the data).
#
#   sbatch scripts/utils/run_data_test.sh   or   bash scripts/utils/run_data_test.sh
#
# Run from the repository root and edit the settings block below.

#SBATCH --job-name=test_data
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x_%A.out


DATA_DIR=./data
OUTPUT_DIR=./output/data_test
CONFIG=config/classification.yaml    # the config of the task: classification, depth_estimation, semantic_segmentation, box_tracking
DATASET=CAER
EXTRA_ARGS="--num_classes 7"         # classification: --num_classes N; box tracking: --max_queries N

mkdir -p logs

# activate your environment here, e.g.:  conda activate video-peft

python main.py \
    --config $CONFIG \
    --state_type test_data \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --dataset_name $DATASET \
    $EXTRA_ARGS
