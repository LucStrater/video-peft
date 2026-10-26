#!/bin/bash
# Train and test on semantic segmentation (VSPW).
#
#   sbatch scripts/semantic_segmentation.sh          # SLURM: seeds 0-4, one per array task
#   SEED=0 bash scripts/semantic_segmentation.sh     # without SLURM: one seed on all local GPUs
#
# Run from the repository root. Edit the settings block below; the defaults are the
# attentive probe from Table 3. The selected settings of every method and dataset
# in the paper (Table 3, Figure 2) are listed in config/hparams.yaml.

#SBATCH --job-name=semantic_segmentation
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2          # one task per GPU
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --array=0-4                  # seeds
#SBATCH --output=logs/%x_%A_%a.out

DATA_DIR=./data                      # contains <DATASET>/videos/, <DATASET>/labels/ and <DATASET>/{train,val,test}.txt
CHECKPOINT=./checkpoints/internvideo_next_base/checkpoint.ckpt   # Pretrained checkpoints

DATASET=vspw_v2
LR=0.0025                            # swept over {5e-4, 1e-3, 2.5e-3, 5e-3}

# Adaptation method (PEFT and/or probe); leave empty for the attentive probe on a frozen backbone.
# Examples (ranks / prompt counts / lrs per dataset are in config/hparams.yaml):
#   LoRA              --peft_type lora --peft_rank 128 --peft_dropout 0.1
#   AdaptFormer       --peft_type adaptformer_adapter --peft_rank 256
#   BPT               --peft_type bpt_deep --n_prompt_tokens 256 --bpt_rank 100
#   full finetuning   --peft_type full_finetune --probe_pooler_type linear --depth_patch_size 14 --depth_tubelet_size 1
#   other probes      --probe_pooler_type dist | dpt | vda   (combine with a PEFT if wanted)
#   linear probe      --probe_pooler_type linear --depth_patch_size 14 --depth_tubelet_size 1
#                     (the linear probe cannot resample, so its output grid must match the
#                     backbone's patch size and tubelet size)
# Temporal context (Section 5.4/5.5): 16 input frames, backbone/PEFT/probe each on 2-frame segments
#                     --clip_len 16 --backbone_clip_len 2 --peft_clip_len 2 --probe_clip_len 2
# Per-frame attentive probe (Figure 2):  --probe_clip_len 1 --depth_tubelet_size 1
METHOD_ARGS=""
RUN_NAME=attentive_probe             # used for the wandb run name

SEED=${SEED:-${SLURM_ARRAY_TASK_ID:-0}}
LAUNCH=""; [ -n "$SLURM_JOB_ID" ] && LAUNCH="srun"   # srun starts one process per GPU under SLURM
mkdir -p logs

# activate your environment here, e.g.:  conda activate video-peft

$LAUNCH python main.py \
    --config config/semantic_segmentation.yaml \
    --data_dir $DATA_DIR \
    --checkpoint_path $CHECKPOINT \
    --dataset_name $DATASET \
    --lr $LR \
    --seed $SEED \
    --wandb_name ${RUN_NAME}_${DATASET}_seed${SEED} \
    $METHOD_ARGS
