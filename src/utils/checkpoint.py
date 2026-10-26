# basic
import os
import torch
import sys

# lightning
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from transformers import SiglipVisionModel, CLIPVisionModel, DINOv3ViTModel, VJEPA2Model, Siglip2VisionModel

# model
from src.models.model_module import VideoModel

@rank_zero_only
def save_checkpoint(args):
    cfg = None  # populated for models that store architecture config in the ckpt

    # load checkpoint
    if (args.model_type == 'video_swin') and (args.run_type == 'finetune'):
        state_dict = inflate_swin_weights_2d_to_3d(args)
    elif (args.model_type == 'video_mae') and ('finetuned' in args.checkpoint_url):
        state_dict = torch.load(args.checkpoint_url, map_location='cpu', weights_only=True)['module']
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    elif (args.model_type == 'video_mae'):
        state_dict = torch.load(args.checkpoint_url, map_location='cpu', weights_only=True)['model']
        state_dict = extract_mae_encoder_weights(state_dict)
    elif (args.model_type == 'vjepa2') or (args.model_type == 'vjepa2_1'):
        state_dict = torch.load(args.checkpoint_url, map_location='cpu', weights_only=True)['encoder']
        state_dict = extract_vjepa2_encoder_weights(state_dict)
    elif (args.model_type == 'internvideo_next'):
        state_dict = torch.load(args.checkpoint_url, map_location='cpu', weights_only=True)
        state_dict = extract_internvideo_next_encoder_weights(state_dict)
    elif (args.model_type == 'dinov3'):
        state_dict = torch.load(args.checkpoint_url, map_location='cpu', weights_only=True)
    elif args.model_type == 'siglip2':
        from huggingface_hub import hf_hub_download
        import safetensors.torch as st
        from transformers import Siglip2VisionConfig
        model_id = f"google/siglip2-{args.model_size}-patch16-256"
        # Config only — avoids the Conv2D/Linear shape mismatch that from_pretrained
        # raises (ckpt stores patch_embedding as [D,3,p,p] but HF class expects [D,3*p*p]).
        hf_config = Siglip2VisionConfig.from_pretrained(model_id)
        cfg = {
            'hidden_size': hf_config.hidden_size,
            'num_attention_heads': hf_config.num_attention_heads,
            'num_hidden_layers': hf_config.num_hidden_layers,
            'intermediate_size': hf_config.intermediate_size,
            'layer_norm_eps': hf_config.layer_norm_eps,
            'attention_dropout': hf_config.attention_dropout,
            'image_size': hf_config.image_size,
            'patch_size': hf_config.patch_size,
        }
        model_file = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        raw_sd = st.load_file(model_file)
        state_dict = extract_siglip2_vision_weights(raw_sd)
        del raw_sd
    elif args.model_type == 'hf_model':
        if "siglip" in args.model_name:
            model_pretrained = SiglipVisionModel.from_pretrained(args.model_name)
        elif "clip" in args.model_name:
            model_pretrained = CLIPVisionModel.from_pretrained(args.model_name)
        elif "dinov3" in args.model_name:
            model_pretrained = DINOv3ViTModel.from_pretrained(args.model_name, token=args.hf_token)
        elif "vjepa2" in args.model_name:
            model_pretrained = VJEPA2Model.from_pretrained(args.model_name)
        else:
            raise ValueError(f"Model {args.model_name} not supported")
        state_dict = model_pretrained.state_dict()
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
        del model_pretrained
    else:
        state_dict = torch.hub.load_state_dict_from_url(args.checkpoint_url, map_location='cpu')['state_dict']

    # create a lightning checkpoint
    lightning_checkpoint = {
        'state_dict': state_dict,
        'pytorch-lightning_version': pl.__version__
    }
    if cfg is not None:
        lightning_checkpoint['cfg'] = cfg

    # create the directory
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)

    # save the checkpoint as a Lightning checkpoint
    torch.save(lightning_checkpoint, args.checkpoint_path)

def extract_siglip2_vision_weights(hf_state_dict):
    """
    Remap HuggingFace Siglip2VisionModel state dict to our backbone layout.

    Key changes:
      vision_model.embeddings.patch_embedding.*  -> model.patch_embedding.*
      vision_model.embeddings.position_embedding.* -> model.position_embedding.*
      vision_model.encoder.layers.*              -> model.layers.*
      vision_model.post_layernorm.*              -> model.post_layernorm.*
      vision_model.head.*                        -> skipped
    patch_embedding.weight is reshaped from Conv2D [D, 3, p, p] to Linear [D, 3*p*p].
    """
    new_sd = {}
    for k, v in hf_state_dict.items():
        if not k.startswith('vision_model.') or k.startswith('vision_model.head.'):
            continue
        k = k[len('vision_model.'):]
        k = k.replace('embeddings.patch_embedding.', 'patch_embedding.')
        k = k.replace('embeddings.position_embedding.', 'position_embedding.')
        k = k.replace('encoder.layers.', 'layers.')
        new_sd[k] = v
    return new_sd

def extract_mae_encoder_weights(mae_state_dict):
    """
    Extracts encoder weights from a MAE model checkpoint.
    Args:
        mae_state_dict (dict): State dict from a MAE model (keys include `encoder.`).

    Returns:
        dict: New state dict compatible with a standard ViT model.
    """
    vit_state_dict = {}
    for key, value in mae_state_dict.items():
        if key.startswith("encoder."):
            new_key = key.replace("encoder.", "", 1)
            vit_state_dict[new_key] = value
        elif key.startswith("patch_embed."):
            vit_state_dict[key] = value
        else:
            continue
    return vit_state_dict

def extract_vjepa2_encoder_weights(mae_state_dict):
    """
    Extracts encoder weights from a MAE model checkpoint.
    Args:
        mae_state_dict (dict): State dict from a MAE model (keys include `encoder.`).

    Returns:
        dict: New state dict compatible with a standard ViT model.
    """
    vit_state_dict = {}
    for key, value in mae_state_dict.items():
        if key.startswith("module.backbone."):
            new_key = key.replace("module.backbone.", "", 1)
            vit_state_dict[new_key] = value
        else:
            continue
    return vit_state_dict

def extract_internvideo_next_encoder_weights(mae_state_dict):
    """
    Extracts encoder weights from a MAE model checkpoint.
    Args:
        mae_state_dict (dict): State dict from a MAE model (keys include `encoder.`).

    Returns:
        dict: New state dict compatible with a standard ViT model.
    """
    vit_state_dict = {}
    for key, value in mae_state_dict.items():
        if key.startswith("model.clip_projector."):
            continue
        elif key.startswith("model."):
            new_key = key.replace("model.", "", 1)
            vit_state_dict[new_key] = value
        else:
            continue
    return vit_state_dict

# Checkpoint Loading
def inflate_swin_weights_2d_to_3d(args):
    """
    Inflates a 2D Swin Transformer state dictionary to a 3D Swin Transformer state dictionary.

    Args:
        args (argparse.Namespace): The arguments namespace.
        state_dict (dict): State dictionary of the 2D Swin Transformer.

    Returns:
        dict: The inflated 3D state dictionary.
    """

    checkpoint = torch.hub.load_state_dict_from_url(args.checkpoint_url, map_location='cpu')
    state_dict = {f"backbone.{k}": v for k, v in checkpoint['model'].items()}

    model = VideoModel(args).model

    # delete relative_position_index
    relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # delete head
    del state_dict["backbone.head.weight"]
    del state_dict["backbone.head.bias"]

    state_dict['backbone.patch_embed.proj.weight'] = state_dict['backbone.patch_embed.proj.weight'].unsqueeze(2).repeat(1,1,args.patch_size[0],1,1) / args.patch_size[0]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        L2 = (2*args.window_size[1]-1) * (2*args.window_size[2]-1)
        wd = args.window_size[0]
        if nH1 != nH2:
            print(f"Error in loading {k}, passing")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(2*args.window_size[1]-1, 2*args.window_size[2]-1),
                    mode='bicubic')
                relative_position_bias_table_pretrained = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)
        state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1,1)

    print("State dict keys:", state_dict.keys())
    print("Model state dict keys:", model.state_dict().keys())

    assert all(key in model.state_dict().keys() for key in state_dict.keys()), "Mismatch between loaded state_dict keys and current model state_dict keys"

    return state_dict
