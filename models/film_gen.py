""" Generate gamma and beta from context for FiLM modulation"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLMGenerator(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) layer.
    Learns to modulate features with affine transformations based on conditioning.
    """
    def __init__(self, cond_dim, feat_dim, gamma_shift=1.0):
        super().__init__()
        self.gamma_shift = gamma_shift

        # Generate gamma (multiplicative) and beta (additive) from conditioning
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, feat_dim * 2) # * 2 for gamma and beta
        )
    
    def get_gamma_beta(self, cond):
        """
        Extract gamma and beta.
        """
        params = self.mlp(cond)  # (B, feat_dim * 2)
        gamma, beta = torch.split(params, params.size(-1) // 2, dim=-1)  # (B, feat_dim) each
        gamma = gamma + self.gamma_shift
        
        return gamma, beta
    
    def apply_film(self, x, gamma, beta):
        """
        Apply already computed gamma and beta to input.
        """
        return gamma * x + beta
    
    def forward(self, x, cond):
        """
        x: Predicted action by diffusion model
        cond: Fused VLA token
        returns: Modulated features
        """
        gamma, beta = self.get_gamma_beta(cond)
        
        # FiLM: y = gamma * x + beta
        return self.apply_film(x, gamma, beta)
