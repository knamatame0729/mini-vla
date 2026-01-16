"""
Train only the FiLM layer while freezing the rest of the VLA model.

Loss: Given action_A (from instruction_A) and instruction_B,
      the FiLM layer should transform action_A into action_B.
      
      FiLM(action_A, text_B) → action_B
"""

import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models.vla_diffusion_policy import VLADiffusionPolicy

import wandb


class FiLMTrainingDataset(Dataset):
    """Dataset for FiLM layer training.
    
    Creates pairs of (action_A, instruction_B) -> action_B for training.
    Groups samples by task_id to create cross-instruction pairs.
    """
    def __init__(self, path, resize_to=64):
        data = np.load(path, allow_pickle=True)
        self.images = data["images"]             # (N, H, W, 3)
        self.states = data["states"]             # (N, state_dim)
        self.actions = data["actions"]           # (N, action_dim)
        self.text_ids = data["text_ids"]         # (N, T_text)
        self.task_ids = data["task_ids"]         # (N,) task identifiers
        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]
        self.resize_to = resize_to
        
        # Group indices by task_id
        self._group_by_task()
        
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None
    
    def _group_by_task(self):
        """Group sample indices by their task_id."""
        self.task_groups = {}
        for idx in range(len(self.task_ids)):
            task_id = str(self.task_ids[idx])
            if task_id not in self.task_groups:
                self.task_groups[task_id] = []
            self.task_groups[task_id].append(idx)
        
        self.task_keys = list(self.task_groups.keys())
        print(f"Found {len(self.task_keys)} unique tasks")
        for key in self.task_keys:
            print(f"  Task '{key}': {len(self.task_groups[key])} samples")

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        """
        Returns a sample with:
        - image, state (from idx for context)
        - action_A: action at idx (from task A)
        - text_ids_A: instruction A
        - action_B: action from a different task B
        - text_ids_B: instruction B
        """
        # Get current sample (task A)
        img = self.images[idx]
        if self.cv2 is not None and (img.shape[0] != self.resize_to or img.shape[1] != self.resize_to):
            img = self.cv2.resize(img, (self.resize_to, self.resize_to))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        state = torch.from_numpy(self.states[idx]).float()
        action_A = torch.from_numpy(self.actions[idx]).float()
        text_ids_A = torch.from_numpy(self.text_ids[idx]).long()
        
        # Get task_id for current sample
        task_A = str(self.task_ids[idx])
        
        # Pick a different task (task B)
        other_tasks = [k for k in self.task_keys if k != task_A]
        if len(other_tasks) == 0:
            # Only one task, use same
            task_B = task_A
        else:
            task_B = other_tasks[np.random.randint(len(other_tasks))]
        
        # Sample a random action from task B
        indices_B = self.task_groups[task_B]
        idx_B = indices_B[np.random.randint(len(indices_B))]
        
        action_B = torch.from_numpy(self.actions[idx_B]).float()
        text_ids_B = torch.from_numpy(self.text_ids[idx_B]).long()
        
        return img, state, action_A, text_ids_A, action_B, text_ids_B


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str,
                        default="data/metaworld_push_and_right_bc.npz")
    parser.add_argument("--pretrained-path", type=str,
                        default="checkpoints/model.pt",
                        help="Path to pretrained VLA model")
    parser.add_argument("--resize-to", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/film_model.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    return parser.parse_args()


def freeze_model_except_film(model):
    """Freeze all parameters except FiLM layer."""
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze FiLM layer in VLADiffusionPolicy
    for param in model.film.parameters():
        param.requires_grad = True
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters (FiLM only): {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return model


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = FiLMTrainingDataset(args.dataset_path, resize_to=args.resize_to)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    
    # Infer model dimensions
    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]
    
    # Initialize model
    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=128,
        diffusion_T=16
    ).to(device)
    
    # Load pretrained weights if available
    if os.path.exists(args.pretrained_path):
        print(f"Loading pretrained model from {args.pretrained_path}")
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print("Pretrained model loaded successfully")
    else:
        print(f"Warning: No pretrained model found at {args.pretrained_path}")
        print("Training FiLM layer from scratch (not recommended)")
    
    # Freeze all except FiLM layer
    model = freeze_model_except_film(model)
    
    # Setup optimizer - only optimize FiLM parameters
    film_params = list(model.film.parameters())
    optimizer = torch.optim.Adam(film_params, lr=args.lr)
    
    # Initialize wandb
    if args.wandb:
        wandb.init(
            project="mini-vla-film",
            config={
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "lr": args.lr,
                "dataset": args.dataset_path,
            }
        )
    
    print(f"\nStarting FiLM layer training for {args.epochs} epochs...")
    print(f"Loss: FiLM(action_A, text_B) -> action_B\n")
    
    best_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        for batch in loader:
            img, state, action_A, text_ids_A, action_B, text_ids_B = batch
            
            img = img.to(device)
            state = state.to(device)
            action_A = action_A.to(device)
            text_ids_A = text_ids_A.to(device)
            action_B = action_B.to(device)
            text_ids_B = text_ids_B.to(device)
            
            optimizer.zero_grad()
            
            # Apply FiLM: transform action_A with instruction_B -> should produce action_B
            # filmed_action returns (new_action, gamma, beta)
            predicted_action_B, gamma, beta = model.filmed_action(action_A, text_ids_B)
            
            # Loss: predicted action should match target action_B
            loss = F.mse_loss(predicted_action_B, action_B)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Logging
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.6f}")
        
        if args.wandb:
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "gamma_mean": gamma.mean().item(),
                "gamma_std": gamma.std().item(),
                "beta_mean": beta.mean().item(),
                "beta_std": beta.std().item(),
            })
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "vocab": dataset.vocab,
            }, args.save_path)
            print(f"  -> Saved best model (loss: {best_loss:.6f})")
    
    print(f"\nTraining complete! Best loss: {best_loss:.6f}")
    print(f"Model saved to {args.save_path}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
