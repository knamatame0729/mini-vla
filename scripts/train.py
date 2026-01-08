"""Train VLA on dataset of image, state, action, and text instruction"""

import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.vla_diffusion_policy import VLADiffusionPolicy

import wandb

class TrainingDataset(Dataset):
    def __init__(self, path, resize_to=64):
        data = np.load(path, allow_pickle=True)
        self.images = data["images"]             # (N, H, W, 3)
        self.states = data["states"]             # (N, state_dim)
        self.actions = data["actions"]           # (N, action_dim)
        self.text_ids = data["text_ids"]         # (N, T_text)
        self.vocab = data["vocab"].item() if data["vocab"].shape == () else data["vocab"]
        self.resize_to = resize_to

        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            self.cv2 = None

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        img = self.images[idx]  # (H, W, 3), uint8
        if self.cv2 is not None and (img.shape[0] != self.resize_to or img.shape[1] != self.resize_to):
            img = self.cv2.resize(img, (self.resize_to, self.resize_to))

        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        state = torch.from_numpy(self.states[idx]).float()
        action = torch.from_numpy(self.actions[idx]).float()
        text_ids = torch.from_numpy(self.text_ids[idx]).long()
        return img, state, action, text_ids


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=str,
                        default="data/dataset.npz")
    parser.add_argument("--resize-to", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--diffusion-T", type=int, default=16)
    parser.add_argument("--num-text-layers", type=int, default=2,
                        help="Number of Transformer layers for text encoder")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of attention heads in Transformer")
    parser.add_argument("--save-path", type=str,
                        default="checkpoints/model.pt")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda' or 'cpu'")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = TrainingDataset(args.dataset_path, resize_to=args.resize_to)
    vocab_size = max(dataset.vocab.values()) + 1
    state_dim = dataset.states.shape[1]
    action_dim = dataset.actions.shape[1]

    model = VLADiffusionPolicy(
        vocab_size=vocab_size,
        state_dim=state_dim,
        action_dim=action_dim,
        d_model=args.d_model,
        diffusion_T=args.diffusion_T,
        num_text_layers=args.num_text_layers,
        num_heads=args.num_heads
    ).to(device)

    # Calculate and print model parameter size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Component breakdown
    img_encoder_params = sum(p.numel() for p in model.img_encoder.parameters())
    txt_encoder_params = sum(p.numel() for p in model.txt_encoder.parameters())
    state_encoder_params = sum(p.numel() for p in model.state_encoder.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    diffusion_params = sum(p.numel() for p in model.diffusion_head.parameters())
    
    print("=" * 60)
    print("VLA Model Parameter Breakdown:")
    print("=" * 60)
    print(f"Image Encoder:     {img_encoder_params:>10,} parameters")
    print(f"Text Encoder:      {txt_encoder_params:>10,} parameters")
    print(f"State Encoder:     {state_encoder_params:>10,} parameters")
    print(f"Fusion MLP:        {fusion_params:>10,} parameters")
    print(f"Diffusion Head:    {diffusion_params:>10,} parameters")
    print("-" * 60)
    print(f"Total parameters:  {total_params:>10,}")
    print(f"Trainable params:  {trainable_params:>10,}")
    print(f"Model size:        {total_params * 4 / (1024**2):>10.2f} MB")
    print("=" * 60)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    num_epochs = args.epochs

    wandb.init(project="mini-vla", config=args)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        epoch_losses = []

        for batch_idx, (img, state, action, text_ids) in enumerate(loader):
            img = img.to(device)
            state = state.to(device)
            action = action.to(device)
            text_ids = text_ids.to(device)

            # Forward pass
            loss, loss_dict = model.loss(img, text_ids, state, action)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * img.size(0)
            epoch_losses.append(loss.item())

            wandb.log(({
                "batch_loss": loss.item(),
                "epoch": epoch + 1,
                "batch": batch_idx
            }))

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch+1}/{num_epochs}  loss={avg_loss:.4f}")

        wandb.log({
            "epoch_loss": avg_loss,
            "epoch_loss_std": np.std(epoch_losses),
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": dataset.vocab,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "d_model": args.d_model,
            "diffusion_T": args.diffusion_T,
        },
        args.save_path,
    )
    print("Saved checkpoint:", args.save_path)


if __name__ == "__main__":
    main()
