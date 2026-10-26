# Where Do We (Not) Need Temporal Context in Low-Resource Video Task Adaptation?

Code for the NeurIPS 2026 (Evaluations and Datasets track) paper by Luc P.J. Sträter and Hazel Doughty.
Project page: https://lucstrater.com/temporal-context/

The codebase benchmarks parameter-efficient finetuning (PEFT) methods and probes on video backbones, on six
low-resource video tasks (the ScarceVid benchmark), and lets the backbone, PEFT modules and probe each operate on
their own temporal context.

## Contents

| | |
|---|---|
| Backbones | InternVideo-Next, VJEPA2, DINOv3, SigLIP2 |
| PEFT methods | BitFit, LayerNorm, SSF, VPT, VFPT, BPT, AdaptFormer, ST-Adapter, LoRA, DoRA, full finetuning |
| Probes | linear, efficient, attentive, DiST, DPT, VDA |
| Tasks | classification (CAER, NurViD, IndustReal, MammAlps), depth estimation (ScanNet), semantic segmentation (VSPW), box tracking |

```
main.py                    entry point, all command line arguments
config/<task>.yaml         default settings per task type (classification, depth_estimation, semantic_segmentation, box_tracking)
config/hparams.yaml        selected hyperparameters for Table 3 and Figure 2, and the sweep ranges
scripts/<task>.sh          SLURM / bash script per task type
src/                       data loading, backbones, PEFT methods, probes, training loop
scripts/utils/             DNxHR re-encoding, dataset conversion, and SLURM scripts for the system and data tests
utils/                     dataset preparation
tests/                     system test and data test
docs/INSTALL.md            installation and reference output of the system test
docs/DATASET.md            video and annotation format per task, tips for SLURM clusters
```

## Getting started

1. **Install** the environment ([docs/INSTALL.md](docs/INSTALL.md)) and check it with the system test:
   ```bash
   python main.py --state_type test_system --config config/classification.yaml
   ```
2. **Prepare the datasets** ([docs/DATASET.md](docs/DATASET.md)) and check the data loading:
   ```bash
   python main.py --state_type test_data --config config/classification.yaml --data_dir ./data --dataset_name CAER --num_classes 7
   ```
3. **Download the pretrained backbone** (see [Pretrained checkpoints](#pretrained-checkpoints)).
4. **Run an experiment** with the script of the task:
   ```bash
   sbatch scripts/classification.sh           # CAER, NurViD, IndustReal, MammAlps
   sbatch scripts/depth_estimation.sh         # ScanNet
   sbatch scripts/semantic_segmentation.sh    # VSPW
   sbatch scripts/box_tracking.sh             # box tracking, e.g. Waymo Open
   ```

## Running experiments

Every run trains for 4,000 steps, selects the checkpoint with the best validation score, and evaluates it on the
test set. The test metrics are printed at the end of the log and logged to Weights and Biases (`--wandb_mode
disabled` to turn this off).

Each script has a settings block at the top with the dataset, learning rate and the adaptation method. Under
SLURM, the job array runs seeds 0-4 in parallel (`--array=0-4`); without SLURM, `SEED=0 bash scripts/<task>.sh`
runs one seed on all GPUs of the machine. The paper uses 2 H100 GPUs with 16 clips per GPU. The learning rates in
this repository are for a global batch size of 32 (`global_batch_size` in the configs) and are scaled linearly
with the actual global batch size (`batch_size * num_gpus`), e.g. halved on a single GPU.

The method is selected with `METHOD_ARGS`, for example:

| Experiment | `METHOD_ARGS` |
|---|---|
| attentive probe (frozen backbone) | *(empty)* |
| + LoRA | `--peft_type lora --peft_rank 128 --peft_dropout 0.1` |
| + AdaptFormer | `--peft_type adaptformer_adapter --peft_rank 256` |
| full finetuning | `--peft_type full_finetune --probe_pooler_type linear` |
| DiST probe | `--probe_pooler_type dist` |
| DiST probe + LoRA | `--probe_pooler_type dist --peft_type lora --peft_rank 256 --peft_dropout 0.1` |
| other video backbone | `--model_type vjepa2 --model_size large --clip_len 16` (with its checkpoint) |
| image backbone | `--model_type dinov3 --model_size large --probe_add_pos_embed true` (with its checkpoint) |
| distributed temporal context | `--clip_len 16 --backbone_clip_len 16 --peft_clip_len 2 --probe_clip_len 2` |

**Hyperparameters.** [config/hparams.yaml](config/hparams.yaml) lists the learning rate, rank and number of prompt
tokens selected for every method and dataset in Table 3 and Figure 2, together with the sweep ranges they were
selected from. For a new method, we sweep the learning rate over {1e-4, 2.5e-4, 5e-4, 1e-3} for classification
and {5e-4, 1e-3, 2.5e-3, 5e-3} for the dense tasks with a single seed, and report mean and sample standard
deviation over 5 seeds at the best setting.

**Image backbones.** DINOv3 and SigLIP2 encode every frame independently, so their tokens carry no information
about the frame they come from. For these backbones, set `--probe_add_pos_embed true`. This adds a fixed 3D
sin-cos (spatio-temporal) position embedding to the backbone tokens before the probe. Video backbones already
encode time, so the configs leave it off (`probe_add_pos_embed: false`). 

**Temporal context.** `--clip_len` sets the number of input frames, and `--backbone_clip_len`,
`--peft_clip_len` and `--probe_clip_len` the temporal context of each component (each defaults to `--clip_len`).
A component with a shorter context is applied independently to each segment of the input (Section 3 of the
paper). The temporal context experiments (Sections 5.4, 5.5) use 16 input frames and ST-Adapter with rank 128.

## Pretrained checkpoints

The first run converts the original pretrained weights (`--checkpoint_url`, a local file) into a checkpoint for
this codebase (`--checkpoint_path`), and later runs load the converted checkpoint directly. Run one job first
(or `sbatch --array=0 ...`) so that parallel seeds do not convert the weights simultaneously. Finetuned
checkpoints are written to `<checkpoint_path dir>/finetuned/<job id>/`.

## Citation

```bibtex
@inproceedings{strater2026temporal,
  title     = {Where Do We (Not) Need Temporal Context in Low-Resource Video Task Adaptation?},
  author    = {Str{\"a}ter, Luc P.J. and Doughty, Hazel},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

The code is released under the [MIT License](LICENSE). Code adapted from other projects keeps its original
license; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Acknowledgments

This work is supported by the Dutch Research Council (NWO) under a Veni grant (VI.Veni.222.160).
