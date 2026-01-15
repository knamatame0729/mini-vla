"""
Visualization script to understand how FiLM layer changes the actions.

The FiLM (Feature-wise Linear Modulation) layer transforms predicted actions using
an affine transformation (y = gamma * x + beta) where gamma and beta are learned
from the conditioning vector (fused VLA token).
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Ensure we can import from models
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.diffusion_head import FiLMLayer


def visualize_film_effects():
    """Visualize how FiLM layer modulates action predictions."""
    
    # Configuration
    cond_dim = 128      # Conditioning vector dimension (fused VLA token)
    feat_dim = 4        # Action dimension (4D for MetaWorld)
    batch_size = 1
    
    device = torch.device('cpu')
    
    # Create FiLM layer
    film_layer = FiLMLayer(cond_dim=cond_dim, feat_dim=feat_dim).to(device)
    
    # Create sample conditioning vectors (fused VLA tokens)
    cond1 = torch.randn(batch_size, cond_dim, device=device)  # Random conditioning 1
    cond2 = torch.randn(batch_size, cond_dim, device=device)  # Random conditioning 2
    cond3 = torch.randn(batch_size, cond_dim, device=device)  # Random conditioning 3
    
    # Create sample action predictions (before FiLM)
    action_pred = torch.randn(batch_size, feat_dim, device=device)
    
    print("=" * 80)
    print("FiLM LAYER VISUALIZATION: How it Changes Actions")
    print("=" * 80)
    print()
    
    print(f"Configuration:")
    print(f"  - Action dimension (feat_dim):     {feat_dim}")
    print(f"  - Conditioning dimension:          {cond_dim}")
    print()
    
    print(f"Original Action Prediction (before FiLM):")
    print(f"  {action_pred.squeeze().detach().numpy()}")
    print()
    
    # Apply FiLM with different conditioning vectors
    with torch.no_grad():
        action_after_film_1 = film_layer(action_pred, cond1)
        action_after_film_2 = film_layer(action_pred, cond2)
        action_after_film_3 = film_layer(action_pred, cond3)
        
        # Also show the gamma and beta parameters
        params1 = film_layer.mlp(cond1)
        gamma1, beta1 = torch.split(params1, params1.size(-1) // 2, dim=-1)
        
        params2 = film_layer.mlp(cond2)
        gamma2, beta2 = torch.split(params2, params2.size(-1) // 2, dim=-1)
        
        params3 = film_layer.mlp(cond3)
        gamma3, beta3 = torch.split(params3, params3.size(-1) // 2, dim=-1)
    
    print(f"FiLM Layer Modulation Parameters (Conditioning 1):")
    print(f"  Gamma (multiplicative): {gamma1.squeeze().detach().numpy()}")
    print(f"  Beta  (additive):       {beta1.squeeze().detach().numpy()}")
    print(f"  Action after FiLM:      {action_after_film_1.squeeze().detach().numpy()}")
    print(f"  Formula: y = gamma * x + beta")
    print(f"  Change from original:   {(action_after_film_1 - action_pred).squeeze().detach().numpy()}")
    print()
    
    print(f"FiLM Layer Modulation Parameters (Conditioning 2):")
    print(f"  Gamma (multiplicative): {gamma2.squeeze().detach().numpy()}")
    print(f"  Beta  (additive):       {beta2.squeeze().detach().numpy()}")
    print(f"  Action after FiLM:      {action_after_film_2.squeeze().detach().numpy()}")
    print(f"  Change from original:   {(action_after_film_2 - action_pred).squeeze().detach().numpy()}")
    print()
    
    print(f"FiLM Layer Modulation Parameters (Conditioning 3):")
    print(f"  Gamma (multiplicative): {gamma3.squeeze().detach().numpy()}")
    print(f"  Beta  (additive):       {beta3.squeeze().detach().numpy()}")
    print(f"  Action after FiLM:      {action_after_film_3.squeeze().detach().numpy()}")
    print(f"  Change from original:   {(action_after_film_3 - action_pred).squeeze().detach().numpy()}")
    print()
    
    # Compute magnitudes for visualization
    action_pred_np = action_pred.squeeze().detach().numpy()
    action_after_1_np = action_after_film_1.squeeze().detach().numpy()
    action_after_2_np = action_after_film_2.squeeze().detach().numpy()
    action_after_3_np = action_after_film_3.squeeze().detach().numpy()
    
    gamma1_np = gamma1.squeeze().detach().numpy()
    gamma2_np = gamma2.squeeze().detach().numpy()
    gamma3_np = gamma3.squeeze().detach().numpy()
    
    # Create visualization plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('FiLM Layer Effect on Action Predictions', fontsize=16, fontweight='bold')
    
    # Plot 1: Action values before and after FiLM
    ax = axes[0, 0]
    x_pos = np.arange(feat_dim)
    width = 0.2
    ax.bar(x_pos - 1.5*width, action_pred_np, width, label='Before FiLM', alpha=0.8)
    ax.bar(x_pos - 0.5*width, action_after_1_np, width, label='After FiLM (cond1)', alpha=0.8)
    ax.bar(x_pos + 0.5*width, action_after_2_np, width, label='After FiLM (cond2)', alpha=0.8)
    ax.bar(x_pos + 1.5*width, action_after_3_np, width, label='After FiLM (cond3)', alpha=0.8)
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Value')
    ax.set_title('Action Values: Before vs After FiLM Modulation')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'a{i}' for i in range(feat_dim)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: Gamma values (multiplicative scaling)
    ax = axes[0, 1]
    ax.bar(x_pos - width, gamma1_np, width, label='Conditioning 1', alpha=0.8)
    ax.bar(x_pos, gamma2_np, width, label='Conditioning 2', alpha=0.8)
    ax.bar(x_pos + width, gamma3_np, width, label='Conditioning 3', alpha=0.8)
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1, label='No scaling (gamma=1)')
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Gamma Value')
    ax.set_title('FiLM Gamma (Multiplicative): Different Conditionings')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'a{i}' for i in range(feat_dim)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Magnitude of changes
    ax = axes[1, 0]
    change_1 = np.abs(action_after_1_np - action_pred_np)
    change_2 = np.abs(action_after_2_np - action_pred_np)
    change_3 = np.abs(action_after_3_np - action_pred_np)
    ax.bar(x_pos - width, change_1, width, label='Conditioning 1', alpha=0.8)
    ax.bar(x_pos, change_2, width, label='Conditioning 2', alpha=0.8)
    ax.bar(x_pos + width, change_3, width, label='Conditioning 3', alpha=0.8)
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Absolute Change')
    ax.set_title('Magnitude of FiLM Modulation Effect')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'a{i}' for i in range(feat_dim)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Percentage change
    ax = axes[1, 1]
    pct_change_1 = 100 * (action_after_1_np - action_pred_np) / np.maximum(np.abs(action_pred_np), 0.01)
    pct_change_2 = 100 * (action_after_2_np - action_pred_np) / np.maximum(np.abs(action_pred_np), 0.01)
    pct_change_3 = 100 * (action_after_3_np - action_pred_np) / np.maximum(np.abs(action_pred_np), 0.01)
    ax.bar(x_pos - width, pct_change_1, width, label='Conditioning 1', alpha=0.8)
    ax.bar(x_pos, pct_change_2, width, label='Conditioning 2', alpha=0.8)
    ax.bar(x_pos + width, pct_change_3, width, label='Conditioning 3', alpha=0.8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Action Dimension')
    ax.set_ylabel('Percentage Change (%)')
    ax.set_title('Percentage Change in Actions Due to FiLM')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'a{i}' for i in range(feat_dim)])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(__file__).parent.parent / 'videos' / 'film_layer_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to: videos/film_layer_analysis.png")
    plt.close()
    
    # Summary
    print()
    print("=" * 80)
    print("SUMMARY: How FiLM Changes Actions")
    print("=" * 80)
    print()
    print("The FiLM layer applies the transformation: y = gamma * x + beta")
    print()
    print("Where:")
    print("  - x       = Original action prediction from the MLP")
    print("  - gamma   = Multiplicative factor (learned from conditioning)")
    print("  - beta    = Additive bias (learned from conditioning)")
    print("  - y       = Final modulated action")
    print()
    print("Key insights:")
    print("  1. Different conditionings (images, text, state) → Different gamma/beta values")
    print("  2. Gamma can scale up (>1) or down (<1) individual action dimensions")
    print("  3. Beta can shift the actions up or down regardless of their original value")
    print("  4. This allows the model to adapt action outputs based on the task context")
    print()
    print("In the diffusion process:")
    print("  - The ActionDenoiseModel predicts noise to remove from noisy actions")
    print("  - FiLM modulates this noise prediction based on the fused VLA token")
    print("  - This conditions the noise prediction on the task (image + text + state)")
    print()


def show_film_architecture():
    """Show the FiLM layer architecture."""
    print("=" * 80)
    print("FiLM LAYER ARCHITECTURE")
    print("=" * 80)
    print()
    print("FiLMLayer Structure:")
    print("  Input: conditioning vector (B, cond_dim=128)")
    print("  ↓")
    print("  MLP:")
    print("    - Linear(128 → 128)")
    print("    - ReLU activation")
    print("    - Linear(128 → 8)  [output has 2*feat_dim = 2*4 = 8 values]")
    print("  ↓")
    print("  Split output into:")
    print("    - Gamma (B, 4) - multiplicative parameters")
    print("    - Beta  (B, 4) - additive parameters")
    print("  ↓")
    print("  Apply to features:")
    print("    output = gamma * x + beta")
    print()
    print("Integration in ActionDenoiseModel:")
    print("  1. Input: noisy actions (x_t), timestep (t), conditioning (cond)")
    print("  2. Embed timestep using sinusoidal embeddings")
    print("  3. Concatenate: [x_t, t_emb, cond]")
    print("  4. Pass through MLP: predict noise (eps_pred)")
    print("  5. Apply FiLM modulation: eps_pred = film(eps_pred, cond)")
    print("  6. Output: modulated noise prediction")
    print()


if __name__ == "__main__":
    show_film_architecture()
    print()
    visualize_film_effects()
