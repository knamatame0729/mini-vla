"""Print detailed neural network parameter information."""

import argparse
import os
import torch
import numpy as np
from tabulate import tabulate

from models.vla_diffusion_policy import VLADiffusionPolicy


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_layer_params(module, prefix=""):
    """Recursively get parameter info for each layer."""
    params_info = []
    
    for name, param in module.named_parameters(recurse=False):
        full_name = f"{prefix}.{name}" if prefix else name
        params_info.append({
            "name": full_name,
            "shape": tuple(param.shape),
            "params": param.numel(),
            "trainable": param.requires_grad,
            "dtype": str(param.dtype).replace("torch.", "")
        })
    
    for name, child in module.named_children():
        child_prefix = f"{prefix}.{name}" if prefix else name
        params_info.extend(get_layer_params(child, child_prefix))
    
    return params_info


def print_component_summary(model):
    """Print parameter summary for each major component."""
    print("\n" + "=" * 80)
    print("MODEL COMPONENT SUMMARY")
    print("=" * 80)
    
    components = [
        ("Image Encoder", model.img_encoder),
        ("Text Encoder", model.txt_encoder),
        ("State Encoder", model.state_encoder),
        ("Fusion MLP", model.fusion),
        ("Diffusion Head", model.diffusion_head),
    ]
    
    table_data = []
    for comp_name, comp_module in components:
        total, trainable = count_parameters(comp_module)
        size_mb = total * 4 / (1024**2)
        table_data.append([
            comp_name,
            f"{total:,}",
            f"{trainable:,}",
            f"{size_mb:.2f} MB"
        ])
    
    print(tabulate(table_data, 
                   headers=["Component", "Total Params", "Trainable Params", "Size"],
                   tablefmt="grid"))
    
    total, trainable = count_parameters(model)
    total_size_mb = total * 4 / (1024**2)
    print("\n" + "-" * 80)
    print(f"TOTAL MODEL:      {total:>15,} parameters")
    print(f"Trainable:        {trainable:>15,} parameters")
    print(f"Non-trainable:    {total - trainable:>15,} parameters")
    print(f"Model size:       {total_size_mb:>15.2f} MB (float32)")
    print("=" * 80)


def print_detailed_layers(model, max_layers=None):
    """Print detailed information about each layer."""
    print("\n" + "=" * 80)
    print("DETAILED LAYER INFORMATION")
    print("=" * 80)
    
    params_info = get_layer_params(model)
    
    if max_layers and len(params_info) > max_layers:
        print(f"\nShowing first {max_layers} layers (use --show-all to see all {len(params_info)} layers)\n")
        params_info = params_info[:max_layers]
    
    table_data = []
    for info in params_info:
        table_data.append([
            info["name"],
            str(info["shape"]),
            f"{info['params']:,}",
            "✓" if info["trainable"] else "✗",
            info["dtype"]
        ])
    
    print(tabulate(table_data,
                   headers=["Layer Name", "Shape", "Parameters", "Trainable", "DType"],
                   tablefmt="grid"))


def print_architecture_summary(model):
    """Print a hierarchical view of the model architecture."""
    print("\n" + "=" * 80)
    print("MODEL ARCHITECTURE")
    print("=" * 80)
    print(model)
    print("=" * 80)


def load_model_from_checkpoint(checkpoint_path, device):
    """Load model from checkpoint file."""
    print(f"\nLoading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata if available
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        metadata = {
            'epoch': checkpoint.get('epoch', 'N/A'),
            'loss': checkpoint.get('loss', 'N/A'),
        }
        print(f"Checkpoint epoch: {metadata['epoch']}, loss: {metadata['loss']}")
    else:
        state_dict = checkpoint
    
    return state_dict


def create_model_from_data(dataset_path, args):
    """Create model based on dataset dimensions."""
    data = np.load(dataset_path, allow_pickle=True)
    vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]
    vocab_size = max(vocab.values()) + 1
    state_dim = data["states"].shape[1]
    action_dim = data["actions"].shape[1]
    
    print(f"\nModel configuration:")
    print(f"  vocab_size: {vocab_size}")
    print(f"  state_dim: {state_dim}")
    print(f"  action_dim: {action_dim}")
    print(f"  d_model: {args.d_model}")
    print(f"  diffusion_T: {args.diffusion_T}")
    
    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        diffusion_T=args.diffusion_T
    )
    
    return model


def main():
    parser = argparse.ArgumentParser(description="Print neural network parameter information")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (e.g., checkpoints/model.pt)")
    parser.add_argument("--dataset-path", type=str, default="data/metaworld_push_bc.npz",
                        help="Path to dataset (needed to infer model dims)")
    parser.add_argument("--d-model", type=int, default=128,
                        help="Model hidden dimension")
    parser.add_argument("--diffusion-T", type=int, default=16,
                        help="Diffusion timesteps")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use (cpu or cuda)")
    parser.add_argument("--show-all", action="store_true",
                        help="Show all layers (default: show first 50)")
    parser.add_argument("--detailed", action="store_true",
                        help="Show detailed layer-by-layer information")
    parser.add_argument("--architecture", action="store_true",
                        help="Show model architecture")
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = create_model_from_data(args.dataset_path, args)
    
    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = load_model_from_checkpoint(args.checkpoint, device)
        model.load_state_dict(state_dict)
    elif args.checkpoint:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random initialization.")
    
    model = model.to(device)
    model.eval()
    
    # Print summaries
    print_component_summary(model)
    
    if args.architecture:
        print_architecture_summary(model)
    
    if args.detailed:
        max_layers = None if args.show_all else 50
        print_detailed_layers(model, max_layers=max_layers)


if __name__ == "__main__":
    main()
