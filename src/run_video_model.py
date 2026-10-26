"""
run_video_model.py

This script is responsible for setting up and running the training and evaluation of video models.
It handles the initialization of the Weights and Biases (wandb) logger, device configuration, 
seeding for reproducibility, and the PyTorch Lightning trainer setup. 

Author: Luc Sträter
"""

# basic
import os
import torch
import sys

# lightning
import lightning.pytorch as pl

# modules
from src.data.data_module import VideoDataModule
from src.models.model_module import VideoModel

# logging
from src.utils.logging import init_wandb, finish_wandb, print_rank_zero
from src.utils.checkpoint import save_checkpoint

def run(args):
    """
    Runs the training and/or evaluation of the video model.

    Args:
        args: Argument parser object.
    """
    # wandb
    wandb_logger = init_wandb(args)
    experiment_id = os.environ.get("SLURM_JOB_ID", "no-slurm")
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    rank = int(os.environ.get("SLURM_PROCID", 0))

    # device
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        sys.exit()
    else:
        num_gpus = int(os.environ.get("SLURM_GPUS_ON_NODE", torch.cuda.device_count()))

    if args.state_type == 'train_model':
        args.lr = args.lr * args.batch_size * num_gpus * num_nodes / args.global_batch_size
        args.min_lr = args.min_lr * args.batch_size * num_gpus * num_nodes / args.global_batch_size
        args.warmup_lr = args.warmup_lr * args.batch_size * num_gpus * num_nodes / args.global_batch_size

    if args.task_type == "depth_estimation":
        monitor = "val_delta1"
    elif args.task_type == "semantic_segmentation":
        monitor = "val_mIoU"
    elif args.task_type == "box_tracking":
        monitor = "val_miou"
    else:
        monitor = "val_top1_acc"

    # lightning trainer
    strategy = "ddp" if num_gpus > 1 else "auto"
    torch.set_float32_matmul_precision('high')

    if args.max_steps > 0:
        duration_kwargs = {
            "max_steps": args.max_steps,
            "val_check_interval": args.val_check_interval,
            "check_val_every_n_epoch": None,
        }
    else:
        duration_kwargs = {
            "max_epochs": args.max_epochs,
            "check_val_every_n_epoch": args.check_val_every_n_epoch,
        }

    trainer = pl.Trainer(
        logger=wandb_logger,
        accelerator="gpu",
        devices=num_gpus,
        strategy=strategy,
        num_nodes=num_nodes,
        reload_dataloaders_every_n_epochs=args.jest_warmup_epochs,
        log_every_n_steps=args.log_every_n_steps,
        enable_progress_bar=False,
        precision='16-mixed' if args.fp16 else "32",
        profiler=pl.profilers.SimpleProfiler(dirpath=args.output_dir, filename="simple_profiler", extended=True) if args.run_profiling else None,
        callbacks=[
            pl.callbacks.ModelCheckpoint(dirpath=os.path.join(os.path.dirname(args.checkpoint_path), "finetuned", experiment_id), filename="checkpoint", save_weights_only=True, mode="max", monitor=monitor),
            pl.callbacks.LearningRateMonitor("step")
        ],
        default_root_dir=args.data_dir,
        **duration_kwargs,
    )

    # seeding
    pl.seed_everything(args.seed + rank)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # initialize the data module
    print_rank_zero("Initializing Video Data Module...")
    data_module = VideoDataModule(args)
    print_rank_zero("Video Data Module initialized successfully.")

    # confirm / create checkpoint path
    if not os.path.exists(args.checkpoint_path):
        print_rank_zero("Loading checkpoint from URL...")
        save_checkpoint(args)
        print_rank_zero("Checkpoint saved successfully.")
    else:
        print_rank_zero("Checkpoint found. Loading from path...")

    # barrier: non-zero ranks must wait for rank 0 to finish writing the checkpoint
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # load model
    print_rank_zero(f"Loading video model: {args.model_type}...")
    model = VideoModel(args)
    print_rank_zero("Video model loaded successfully.")

    # compile
    if args.compile_model:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model)

    # train
    if args.state_type == 'train_model':
        print_rank_zero("Training...")
        trainer.fit(model, data_module)
        checkpoint_dir = trainer.checkpoint_callback.best_model_path
        print_rank_zero(f"Best checkpoint: {checkpoint_dir}")
        model = VideoModel.load_from_checkpoint(checkpoint_dir, args=args)
    elif args.state_type == 'test_model':
        model = VideoModel.load_from_checkpoint(args.checkpoint_path, args=args, strict=False)

    # test  
    print_rank_zero("Testing...")
    trainer.test(model, data_module, verbose=True)
    
    # wandb
    finish_wandb()