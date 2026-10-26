"""
main.py

This script serves as the entry point for running different types of tests and models 
related to video understanding.

Author: Luc P.J. Sträter
"""

import argparse
import yaml

# import functions
from tests.data_test import test_video_data_module
from tests.system_test import test_system
from src.run_video_model import run

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser(description="Video Understanding")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")

    # load config defaults
    config_str = parser.parse_known_args()[0].config
    config = load_config(config_str)   

    # basic
    parser.add_argument("--state_type", default=config.get("state_type", "train_model"), choices=['train_model', 'test_model', 'test_data', 'test_system'], help="The state that has to be run")
    parser.add_argument("--task_type", default=config.get("task_type", "classification"), choices=['classification', 'classification_epic', 'depth_estimation', 'semantic_segmentation', 'box_tracking'], help="The task type that has to be run")
    parser.add_argument("--model_type", default=config.get("model_type", "video_swin"), help="The model that has to be run")
    parser.add_argument("--run_type", default=config.get("run_type", "test"), choices=['pretrain', 'finetune', 'probe'], help="The run type that has to be run")
    parser.add_argument("--seed", type=int, default=config.get("seed", 0), help="Seed for random number generators")
    parser.add_argument("--deterministic", action='store_true', default=config.get("deterministic", True), help="Deterministic mode")

    # data 
    parser.add_argument('--data_dir', type=str, default=config.get('data_dir', './data'), help='Path to the dataset directory')
    parser.add_argument('--dataset_name', type=str, default=config.get('dataset_name', 'Kinetics-400'), help='Name of the dataset')
    parser.add_argument('--annotation_type', type=str, default=config.get('annotation_type', 'json'), choices=['csv', 'json', 'epic'], help='Type of the annotation file')
    parser.add_argument('--video_format', type=str, default=config.get('video_format', 'mov'), choices=['mp4', 'mov', 'avi'], help='Format of the video files')
    parser.add_argument('--output_dir', type=str, default=config.get('output_dir', './output_dir'), help='Path to save the frames')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 8), help='Batch size for training')
    parser.add_argument('--global_batch_size', type=int, default=config.get('global_batch_size', 256), help='Global batch size for training')
    parser.add_argument('--test_batch_size', type=int, default=config.get('test_batch_size', 4), help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=config.get('num_workers', 8), help='Number of workers for data loading')
    parser.add_argument('--clip_len', type=int, default=config.get('clip_len', 32), help='Number of frames per clip (T_data, always the max, what dataloader loads)')
    parser.add_argument('--backbone_clip_len', type=int, default=config.get('backbone_clip_len', None), help='Frames backbone processes per forward (T_model); defaults to clip_len')
    parser.add_argument('--peft_clip_len', type=int, default=config.get('peft_clip_len', None), help='Frames PEFT module sees per forward (T_peft); defaults to backbone_clip_len')
    parser.add_argument('--probe_clip_len', type=int, default=config.get('probe_clip_len', None), help='Frames probe sees per forward (T_probe); defaults to clip_len')
    parser.add_argument('--sampling_rate', type=int, default=config.get('sampling_rate', 2), help='Sampling rate for frames')
    parser.add_argument('--num_clips', type=int, default=config.get('num_clips', 1), help='Number of clips per video for training')
    parser.add_argument('--test_num_clips', type=int, default=config.get('test_num_clips', 2), help='Number of clips per video for testing')
    parser.add_argument('--num_crops', type=int, default=config.get('num_crops', 1), help='Number of crops per frame for training')
    parser.add_argument('--test_num_crops', type=int, default=config.get('test_num_crops', 3), help='Number of crops per frame for testing')
    parser.add_argument('--input_size', type=int, default=config.get('input_size', 224), help='Input size for cropping frames')
    parser.add_argument('--short_side_size', type=int, default=config.get('short_side_size', 256), help='Short side size for resizing frames')
    parser.add_argument('--frame_sampling_method', type=str, default=config.get('frame_sampling_method', 'uniform_segment_sampling'), help='Frame sampling method')
    parser.add_argument('--model_name', type=str, default=config.get('model_name', 'dinov3_base'), help='Model name')
    parser.add_argument('--model_size', type=str, default=config.get('model_size', 'base'), help='Model size')
    parser.add_argument('--reference_model_name', type=str, default=config.get('reference_model_name', 'none'), help='Which model to use for aggregation')
    parser.add_argument("--reference_model_path", type=str, default=config.get("reference_model_path", "/checkpoints/checkpoint.ckpt"), help="Path to small model checkpoint")
    
    # logging
    parser.add_argument("--log_every_n_steps", type=int, default=config.get("log_every_n_steps", 10), help="Log frequency")
    parser.add_argument("--check_val_every_n_epoch", type=int, default=config.get("check_val_every_n_epoch", 1), help="Check validation every n epochs")
    parser.add_argument("--wandb_project", type=str, default=config.get("wandb_project", "video-peft"), help="WandB project")
    parser.add_argument("--wandb_entity", type=str, default=config.get("wandb_entity", None), help="WandB entity (default: your default entity)")
    parser.add_argument("--wandb_name", type=str, default=config.get("wandb_name", "default"), help="WandB run name")
    parser.add_argument("--wandb_mode", type=str, default=config.get("wandb_mode", "online"), choices=['online', 'offline', 'disabled'], help="WandB mode; the API key is read from WANDB_API_KEY or `wandb login`")

    # pooler
    parser.add_argument("--feature_type", type=str, default=config.get("feature_type", "spatial"), choices=['spatial', 'gap', 'cls'], help="Feature type of the backbone, ie spatial, cls, or gap (mean of all spatial tokens)")
    parser.add_argument("--probe_depth", type=int, default=config.get("probe_depth", 1), help="Depth of the attentive pooler")
    parser.add_argument("--probe_pooler_type", type=str, default=config.get("probe_pooler_type", "attentive"), choices=['attentive', 'attentive_multilayer', 'efficient', 'linear', 'dist', 'dpt', 'vda'], help="Pooler type for the probe")
    parser.add_argument("--probe_drop_path_rate", type=float, default=config.get("probe_drop_path_rate", 0.0), help="Stochastic depth rate for the probe")
    parser.add_argument("--probe_use_rope", type=str2bool, default=config.get("probe_use_rope", True), help="Use rope in the attentive pooler")
    parser.add_argument("--probe_add_pos_embed", type=str2bool, default=config.get("probe_add_pos_embed", True), help="Add position embedding in the attentive pooler")

    # classification
    parser.add_argument("--num_classes", type=int, default=config.get("num_classes", 400), help="Number of classes")

    # classification epic
    parser.add_argument('--num_verb_classes', type=int, default=config.get('num_verb_classes', 97), help='Number of verb classes')
    parser.add_argument('--num_noun_classes', type=int, default=config.get('num_noun_classes', 300), help='Number of noun classes')

    # depth estimation
    parser.add_argument('--depth_patch_size', type=int, default=config.get('depth_patch_size', 8), help='Patch size for depth estimation task')
    parser.add_argument('--depth_tubelet_size', type=int, default=config.get('depth_tubelet_size', 2), help='Tubelet size for depth estimation task')
    parser.add_argument('--min_depth', type=float, default=config.get('min_depth', 0.001), help='Minimum depth for depth estimation task')
    parser.add_argument('--max_depth', type=float, default=config.get('max_depth', 10.0), help='Maximum depth for depth estimation task')

    # semantic segmentation
    parser.add_argument('--seg_void_label', type=int, default=config.get('seg_void_label', 255), help='Void label for semantic segmentation task')

    # box tracking
    parser.add_argument('--max_queries', type=int, default=config.get('max_queries', 25), help='Maximum number of tracks queried per clip (Waymo: 25, CholecTrack20: 1)')
    parser.add_argument('--box_area_frac', type=float, default=config.get('box_area_frac', 0.005), help='Minimum box area (fraction of the frame) for a track to be queried at the query frame')
    parser.add_argument('--box_min_visible_frac', type=float, default=config.get('box_min_visible_frac', 0.625), help='Fraction of the clip a track must stay in the crop to be queried (0.625 = 10/16, matches the annotations)')

    # peft
    parser.add_argument("--peft_type", type=str, default=config.get("peft_type", None), help="PEFT type")
    parser.add_argument("--peft_rank", type=int, default=config.get("peft_rank", 0), help="LoRA rank")
    parser.add_argument("--peft_dropout", type=float, default=config.get("peft_dropout", 0.0), help="PEFT dropout")
    parser.add_argument("--peft_unfreeze_norm", type=str2bool, default=config.get("peft_unfreeze_norm", False), help="Unfreeze the norm layers")

    # prompt tokens
    parser.add_argument("--n_prompt_tokens", type=int, default=config.get("n_prompt_tokens", 0), help="Number of prompt tokens")
    parser.add_argument("--prompt_tokens_type", type=str, default=config.get("prompt_tokens_type", "none"), help="Prompt tokens type")
    parser.add_argument("--prompt_fourier_ratio", type=float, default=config.get("prompt_fourier_ratio", 0.0), help="Fourier ratio for prompt tokens")
    parser.add_argument("--bpt_rank", type=int, default=config.get("bpt_rank", -1), help="BPT rank, -1 means standard VPT")
    
    # learning
    parser.add_argument("--max_epochs", type=int, default=config.get("max_epochs", 30), help="Number of epochs (ignored when max_steps > 0)")
    parser.add_argument("--max_steps", type=int, default=config.get("max_steps", -1), help="Number of training steps; -1 disables (use max_epochs instead)")
    parser.add_argument("--lr", type=float, default=config.get("lr", 1e-3), help="Learning rate")
    parser.add_argument("--lr_multiplier", type=float, default=config.get("lr_multiplier", 0.1), help="Learning rate multiplier")
    parser.add_argument("--weight_decay", type=float, default=config.get("weight_decay", 0.02), help="Weight decay")
    parser.add_argument("--warmup_factor", type=float, default=config.get("warmup_factor", 0.1), help="Warmup factor")
    parser.add_argument("--warmup_epochs", type=float, default=config.get("warmup_epochs", 3), help="Warmup epochs (ignored when max_steps > 0)")
    parser.add_argument("--warmup_steps", type=int, default=config.get("warmup_steps", 400), help="Warmup steps (used when max_steps > 0)")
    parser.add_argument("--val_check_interval", type=int, default=config.get("val_check_interval", 500), help="Validate every N steps (used when max_steps > 0)")
    parser.add_argument("--warmup_lr", type=float, default=config.get("warmup_lr", 1e-6), help="Warmup learning rate")
    parser.add_argument("--min_lr", type=float, default=config.get("min_lr", 1e-6), help="Minimum learning rate")
    parser.add_argument("--drop_path_rate", type=float, default=config.get("drop_path_rate", 0.0), help="Stochastic depth rate")
    parser.add_argument("--layer_decay", type=float, default=config.get("layer_decay", 0.7), help="Layer decay")
    parser.add_argument("--optimizer_beta_1", type=float, default=config.get("optimizer_beta_1", 0.9), help="Optimizer beta 1")
    parser.add_argument("--optimizer_beta_2", type=float, default=config.get("optimizer_beta_2", 0.999), help="Optimizer beta 2")
    parser.add_argument("--optimizer_epsilon", type=float, default=config.get("optimizer_epsilon", 1e-8), help="Optimizer epsilon")

    # jest
    parser.add_argument('--jest_interval', type=int, default=config.get("jest_interval", 1), help='Interval for JEST, for example 5 would mean 20% of the data is trained on.')
    parser.add_argument('--learnability_reduction', type=str, default=config.get("learnability_reduction", "batch"), choices=['batch', 'sample', 'sample_all'], help='Method to reduce the JEST samples (batch, sample, sample_all)')
    parser.add_argument('--jest_warmup_epochs', type=int, default=config.get("jest_warmup_epochs", 0), help='JEST warmup epochs')
    parser.add_argument('--ema_decay_start', type=float, default=config.get("ema_decay_start", 0.99), help='EMA decay start')	
    parser.add_argument('--ema_decay_end', type=float, default=config.get("ema_decay_end", 0.99), help='EMA decay end')

    # augmentations
    parser.add_argument('--num_samples', type=int, default=config.get("num_samples", 1), help='Number of repeated augmentations per video')
    parser.add_argument('--num_samples_method', type=str, default=config.get("num_samples_method", "spatial"), help='Method to sample the repeated samples in the dataloader (spatial, temporal, spatial_temporal)')
    parser.add_argument('--sample_strategy', type=str, default=config.get("sample_strategy", "repeated_augmentation"), help='Strategy to sample views')
    parser.add_argument('--horizontal_flip', type=str2bool, default=config.get("horizontal_flip", False), help='Horizontal flip enabled if True')
    parser.add_argument('--reprob', type=float, default=config.get("reprob", 0.0), help='Random erase prob (default: 0.0)')    
    parser.add_argument('--auto_augment', type=str2bool, default=config.get("auto_augment", False), help='Auto augment enabled if True')
    parser.add_argument('--motion_shift', type=str2bool, default=config.get("motion_shift", False), help='Motion shift enabled if True')
    parser.add_argument('--min_scale', type=float, default=config.get('min_scale', 0.08), help='Minimum scale for random resized crop')
    parser.add_argument('--max_scale', type=float, default=config.get('max_scale', 1.0), help='Maximum scale for random resized crop')
    parser.add_argument('--min_aspect_ratio', type=float, default=config.get('min_aspect_ratio', 0.75), help='Minimum aspect ratio for random resized crop')
    parser.add_argument('--max_aspect_ratio', type=float, default=config.get('max_aspect_ratio', 4./3.), help='Maximum aspect ratio for random resized crop')

    # mixup
    parser.add_argument('--mixup_enabled', type=str2bool, default=config.get("mixup_enabled", False), help='Mixup enabled if True')
    parser.add_argument('--mixup', type=float, default=config.get("mixup", 0.8), help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=config.get("cutmix", 1.0), help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=config.get("cutmix_minmax", None), help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=config.get("mixup_prob", 1.0), help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=config.get("mixup_switch_prob", 0.5), help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default=config.get("mixup_mode", "batch"), help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')    
    parser.add_argument("--smoothing", type=float, default=config.get("smoothing", 0.1), help="Smoothing for label smoothing")
    
    # efficiency
    parser.add_argument("--compile_model", action='store_true', default=config.get("compile_model", False), help="Compile the model")
    parser.add_argument("--fp16", type=str2bool, default=config.get("fp16", False), help="Use FP16 precision")
    parser.add_argument("--run_profiling", action='store_true', default=config.get("run_profiling", False), help="Run profiling")

    # checkpoint
    parser.add_argument("--checkpoint_path", type=str, default=config.get("checkpoint_path"), help="Path to checkpoint")
    parser.add_argument("--checkpoint_url", type=str, default=config.get("checkpoint_url"), help="URL to checkpoint")

    # temporary
    parser.add_argument("--debug_str", type=str, default=config.get("debug_str", ""), help="Debug string")

    # parse arguments
    args = parser.parse_args()

    # resolve temporal-decoupling defaults (each tier defaults to its parent)
    if args.backbone_clip_len is None:
        args.backbone_clip_len = args.clip_len
    if args.peft_clip_len is None:
        args.peft_clip_len = args.backbone_clip_len
    if args.probe_clip_len is None:
        args.probe_clip_len = args.clip_len

    # clip_len must cover all component contexts
    args.clip_len = max(args.clip_len, args.backbone_clip_len, args.peft_clip_len, args.probe_clip_len)

    return args

def main(args):
    if (args.state_type == 'train_model') or (args.state_type == 'test_model'):
        run(args)
    elif args.state_type == 'test_data':
        print("Running data test...")
        test_video_data_module(args)
    elif args.state_type == 'test_system':
        print("Running system test...")
        test_system()

if __name__ == "__main__":
    # parse arguments
    args = get_args()

    # run main
    main(args)
