"""
data_test.py

This file contains functions and utilities for testing data loading.

Author: Luc Sträter
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from lightning.pytorch import seed_everything
from src.data.data_module import VideoDataModule
import time

def save_frames(frames, output_dir, prefix, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Saves frames to a single figure in the specified directory after unnormalizing.

    Args:
        frames (torch.Tensor): Tensor of shape (C, T, H, W).
        output_dir (str): Directory where frames will be saved.
        prefix (str): Prefix for the saved figure filename.
        mean (list): Mean used for normalization (default: ImageNet means).
        std (list): Standard deviation used for normalization (default: ImageNet stds).
    """
    os.makedirs(output_dir, exist_ok=True)

    # unnormalize the frames
    mean = torch.tensor(mean).view(-1, 1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1, 1)
    frames = frames * std + mean 

    # permute and move to cpu
    frames = frames.permute(1, 2, 3, 0).cpu().numpy()  
    frames = np.clip(frames, 0, 1)  

    num_frames = frames.shape[0]
    cols = min(num_frames, 4)
    rows = (num_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()

    for i in range(rows * cols):
        if i < num_frames:
            frame = (frames[i] * 255).astype(np.uint8)  
            axes[i].imshow(frame, aspect='equal')
            axes[i].axis('off')
            axes[i].set_title(f"Frame {i+1}")
        else:
            axes[i].axis('off')

    plt.tight_layout()
    figure_path = os.path.join(output_dir, f"{prefix}_frames.png")
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {figure_path}")

def save_frames_with_boxes(frames, boxes, valid, output_dir, prefix, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Saves frames with their tracked boxes drawn on top, to check the boxes line up with
    the pixels after the video transform.

    Args:
        frames (torch.Tensor): Tensor of shape (C, T, H, W).
        boxes (torch.Tensor): Tensor of shape (max_queries, T, 4) in xyxy pixel coordinates.
        valid (torch.Tensor): Bool tensor of shape (max_queries, T).
        output_dir (str): Directory where frames will be saved.
        prefix (str): Prefix for the saved figure filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    mean = torch.tensor(mean).view(-1, 1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1, 1)
    frames = frames * std + mean

    frames = frames.permute(1, 2, 3, 0).cpu().numpy()
    frames = np.clip(frames, 0, 1)
    boxes = boxes.cpu().numpy()
    valid = valid.cpu().numpy()

    num_frames = frames.shape[0]
    cols = min(num_frames, 4)
    rows = (num_frames + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    axes = axes.flatten()

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(boxes), 1)))

    for i in range(rows * cols):
        if i >= num_frames:
            axes[i].axis('off')
            continue

        axes[i].imshow((frames[i] * 255).astype(np.uint8), aspect='equal')
        axes[i].axis('off')
        axes[i].set_title(f"Frame {i+1}")

        for query_idx in range(len(boxes)):
            if not valid[query_idx, i]:
                continue
            x1, y1, x2, y2 = boxes[query_idx, i]
            axes[i].add_patch(plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=colors[query_idx], linewidth=1.5))

    plt.tight_layout()
    figure_path = os.path.join(output_dir, f"{prefix}_frames.png")
    plt.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figure: {figure_path}")

def save_attention_map(attention_map, output_dir, prefix):
    """
    Saves the attention map to a figure in the specified directory.

    Args:
        attention_map (torch.Tensor): Tensor of the attention map with shape [clip_len, num_H, num_W].
        output_dir (str): Directory where the attention map will be saved.
        prefix (str): Prefix for the saved figure filename.
    """
    os.makedirs(output_dir, exist_ok=True)

    # move to cpu and convert to numpy
    attention_map = attention_map.cpu().numpy()
    
    clip_len, num_H, num_W = attention_map.shape
    
    # Create a figure with subplots for each frame
    fig, axes = plt.subplots(1, clip_len, figsize=(5 * clip_len, 5))
    if clip_len == 1:
        axes = [axes]  # Make axes iterable when there's only one subplot
    
    # Plot each frame's attention map
    for t in range(clip_len):
        cax = axes[t].matshow(attention_map[t], cmap='viridis', aspect='equal')
        axes[t].set_title(f"Frame {t+1}")
        axes[t].axis('off')
    
    # Add a colorbar to the right of the subplots
    fig.colorbar(cax, ax=axes, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{prefix}_attention_map.png"), bbox_inches='tight')
    plt.close(fig)
    print(f"Saved attention map: {os.path.join(output_dir, f'{prefix}_attention_map.png')}")

def flatten_loaders(name, loader):
    """Convert loader(s) to list of (name, loader) tuples"""
    if isinstance(loader, (list, tuple)):
        return [(f"{name}_{idx}", l) for idx, l in enumerate(loader)]
    else:
        return [(name, loader)]

def test_video_data_module(args):
    """
    Tests the videodatamodule by loading batches from train, val, and test loaders
    and saves frames to the specified directory.
    """
    # seeding
    print("Setting seed for reproducibility...")
    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to {args.seed}")

    # initialize the videodatamodule
    print("Initializing VideoDataModule with the provided arguments...")
    data_module = VideoDataModule(args)
    print("VideoDataModule initialized successfully.")

    # Flatten all loaders
    all_loaders = []
    for split_name, loader in [('Train', data_module.train_dataloader()),
                            ('Validation', data_module.val_dataloader()),
                            ('Test', data_module.test_dataloader())]:
        all_loaders.extend(flatten_loaders(split_name, loader))

    # Now iterate normally
    for loader_name, loader in all_loaders:
        print(f"\nTesting {loader_name} dataloader...")

        # check sampling
        num_batches = len(loader)
        total_samples = len(loader.dataset)  
        print(f"Number of batches in {loader_name} loader: {num_batches}")
        print(f"Number of samples in {loader_name} dataset: {total_samples}")

        for batch_idx, (videos, labels) in enumerate(loader):
            print(f"\nBatch {batch_idx + 1}/{num_batches}")
            print(f"Video output shape: {videos.shape}")

            # visualization
            random_idx = 2
            frames = videos[random_idx]  # Shape: (C, T, H, W)

            if args.task_type == 'box_tracking':
                boxes = labels['boxes'][random_idx]  # (max_queries, T, 4)
                valid = labels['valid'][random_idx]  # (max_queries, T)
                print(f"Boxes shape: {boxes.shape}, valid shape: {valid.shape}")
                print(f"Queried tracks per sample: {labels['valid'][:, :, 0].sum(dim=1).tolist()}")
                prefix = f"{loader_name.lower()}_batch{batch_idx + 1}_queries{int(valid[:, 0].sum())}"
                save_frames_with_boxes(frames, boxes, valid, args.output_dir, prefix)
            else:
                print(f"Labels: {labels}")
                label = labels[random_idx]
                prefix = f"{loader_name.lower()}_batch{batch_idx + 1}_label{label}"
                save_frames(frames, args.output_dir, prefix)

            # only one batch
            break

