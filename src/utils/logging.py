"""
logging.py

This file contains utils files for logging.

Author: Luc Sträter
"""

# lightning
import lightning.pytorch as pl
import torch

# logging
import os
import wandb
from wandb.errors import CommError
from lightning.pytorch.utilities.rank_zero import rank_zero_only

@rank_zero_only
def init_wandb(args):
    """
    Initializes Weights and Biases logging only on the main process (rank 0).

    Args:
        args: Argument parser object.
    """
    os.environ["WANDB_RUN_ID"] = wandb.util.generate_id()

    common_kwargs = dict(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f'{args.wandb_name}',
        id=os.environ["WANDB_RUN_ID"],
    )
    try:
        run = wandb.init(**common_kwargs, mode=args.wandb_mode)
    except CommError:
        print("wandb: no connection, falling back to offline mode (sync later with `wandb sync`)")
        run = wandb.init(**common_kwargs, mode="offline")

    return pl.loggers.WandbLogger(experiment=run)

@rank_zero_only
def finish_wandb():
    """
    Finishes Weights and Biases logging only on the main process (rank 0).
    """
    wandb.finish()

@rank_zero_only
def print_rank_zero(x):
    """
    Prints the given message only on the main process (rank 0).
    """
    print(x)


