#!/bin/bash
# Train and test on a box tracking task, e.g. Waymo Open or CholecTrack20.
#
#   sbatch scripts/box_tracking.sh          # SLURM: seeds 0-4, one per array task
#   SEED=0 bash scripts/box_tracking.sh     # without SLURM: one seed on all local GPUs
#
# Run from the repository root and edit the settings block below. The probe is always the attentive
# probe: its queries are computed from the boxes at the query frame.

#SBATCH --job-name=box_tracking
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # one task per GPU
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --array=0-4                  # seeds
#SBATCH --output=logs/%x_%A_%a.out

DATA_DIR=./data                      # contains <DATASET>/videos/ and <DATASET>/{train,val,test}.json
CHECKPOINT=./checkpoints/internvideo_next_base/checkpoint.ckpt   # Pretrained checkpoints

DATASET=waymo_open_iv_v2
MAX_QUERIES=25                       # max. tracks per clip; Waymo Open: 25, CholecTrack20: 1
LR=0.001                             # we sweep over {5e-4, 1e-3, 2.5e-3, 5e-3}

# Backbone and PEFT arguments; leave empty for InternVideo-Next ViT-B with a frozen backbone. Examples:
#   other backbone    --model_type dinov3 --model_size base --probe_add_pos_embed true   (with its checkpoint)
#   AdaptFormer       --peft_type adaptformer_adapter --peft_rank 128
#   LoRA              --peft_type lora --peft_rank 128 --peft_dropout 0.1
METHOD_ARGS=""
RUN_NAME=attentive_probe             # used for the wandb run name

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
LAUNCH=""; [ -n "$SLURM_JOB_ID" ] && LAUNCH="srun"   # srun starts one process per GPU under SLURM
mkdir -p logs

# activate your environment here, e.g.:  conda activate video-peft

$LAUNCH python main.py \
    --config config/box_tracking.yaml \
    --data_dir $DATA_DIR \
    --checkpoint_path $CHECKPOINT \
    --dataset_name $DATASET \
    --max_queries $MAX_QUERIES \
    --lr $LR \
    --seed $SEED \
    --wandb_name ${RUN_NAME}_${DATASET}_seed${SEED} \
    $METHOD_ARGS
