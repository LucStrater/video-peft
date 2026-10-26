#!/bin/bash
# Train and test on a classification task: CAER, NurViD, IndustReal or MammAlps.
#
#   sbatch scripts/classification.sh          # SLURM: seeds 0-4, one per array task
#   SEED=0 bash scripts/classification.sh     # without SLURM: one seed on all local GPUs
#
# Run from the repository root. Edit the settings block below; the defaults are the
# attentive probe on CAER from Table 3. The selected settings of every method and dataset
# in the paper (Table 3, Figure 2) are listed in config/hparams.yaml.

#SBATCH --job-name=classification
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # one task per GPU
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --array=0-4                  # seeds
#SBATCH --output=logs/%x_%A_%a.out

DATA_DIR=./data                      # contains <DATASET>/videos/ and <DATASET>/{train,val,test}.json
CHECKPOINT=./checkpoints/internvideo_next_base/checkpoint.ckpt   # Pretrained checkpoints

DATASET=CAER                         # CAER | NurVid | IndustReal | MammAlps
NUM_CLASSES=7                        # 7    | 51     | 75         | 11
LR=0.00025                           # swept over {1e-4, 2.5e-4, 5e-4, 1e-3}

# Adaptation method (PEFT and/or probe); leave empty for the attentive probe on a frozen backbone.
# Examples (ranks / prompt counts / lrs per dataset are in config/hparams.yaml):
#   LoRA              --peft_type lora --peft_rank 128 --peft_dropout 0.1
#   AdaptFormer       --peft_type adaptformer_adapter --peft_rank 128
#   BPT               --peft_type bpt_deep --n_prompt_tokens 256 --bpt_rank 100
#   full finetuning   --peft_type full_finetune --probe_pooler_type linear
#   other probes      --probe_pooler_type linear | efficient | dist | dpt | vda   (combine with a PEFT if wanted)
# Temporal context (Section 5.4/5.5): 16 input frames, backbone/PEFT/probe each on 2-frame segments
#                     --clip_len 16 --backbone_clip_len 2 --peft_clip_len 2 --probe_clip_len 2
METHOD_ARGS=""
RUN_NAME=attentive_probe             # used for the wandb run name

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
LAUNCH=""; [ -n "$SLURM_JOB_ID" ] && LAUNCH="srun"   # srun starts one process per GPU under SLURM
mkdir -p logs

# activate your environment here, e.g.:  conda activate video-peft

$LAUNCH python main.py \
    --config config/classification.yaml \
    --data_dir $DATA_DIR \
    --checkpoint_path $CHECKPOINT \
    --dataset_name $DATASET \
    --num_classes $NUM_CLASSES \
    --lr $LR \
    --seed $SEED \
    --wandb_name ${RUN_NAME}_${DATASET}_seed${SEED} \
    $METHOD_ARGS
