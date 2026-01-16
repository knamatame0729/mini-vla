"""VLA Diffusion Policy Model."""

import torch
import torch.nn as nn
#from .encoders import ImageEncoderTinyCNN, TextEncoderTransformer, StateEncoderMLP
from .encoders import ImageEncoderTinyCNN, TextEncoderTinyGRU, StateEncoderMLP
from .fusion import FusionMLP
from .diffusion_head import DiffusionConfig, DiffusionPolicyHead
from .film_gen import FiLMGenerator


class VLADiffusionPolicy(nn.Module):
    def __init__(self, vocab_size, state_dim, action_dim,
                 d_model=128, diffusion_T=16):
        super().__init__()
        self.img_encoder = ImageEncoderTinyCNN(d_model=d_model)
        self.txt_encoder = TextEncoderTinyGRU(vocab_size=vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim=state_dim, d_model=d_model)
        self.fusion = FusionMLP(d_model=d_model)
        self.film = FiLMGenerator(cond_dim=d_model, feat_dim=action_dim, gamma_shift=1.0)

        cfg = DiffusionConfig(
            T=diffusion_T,
            action_dim=action_dim,
            cond_dim=d_model,
        )
        self.diffusion_head = DiffusionPolicyHead(cfg)

    def encode_obs(self, image, text_tokens, state):
        img_token = self.img_encoder(image)  # (B, d_model)
        txt_token = self.txt_encoder(text_tokens)  # (B, d_model)
        state_token = self.state_encoder(state)  # (B, d_model)
        fused_context = self.fusion(img_token, txt_token, state_token)
        return fused_context
    
    def encode_text(self, text_tokens):
        """
        Encode text tokens only for generating gamma/beta from new prompt.
        """
        return self.txt_encoder(text_tokens)
    
    def get_gamma_beta_from_text(self, text_tokens):
        """
        Generate gamma and beta values from text tokens using the FiLM layer.
        """
        txt_token = self.encode_text(text_tokens)
        film_layer = self.film
        gamma, beta = film_layer.get_gamma_beta(txt_token)
        return gamma, beta
    
    def apply_film_to_action(self, action, gamma, beta):
        """
        Apply gamma and beta to an action using FiLM formula: y = gamma * x + beta.
        """
        film_layer = self.diffusion_head.denoise_model.film
        return film_layer.apply_film(action, gamma, beta)
    
    def filmed_action(self, action, new_text_tokens):
        """
        action: (B, action_dim) previous action
        new_text_tokens: (B, T_text) new text instruction tokens
        returns: (B, action_dim) filmed action
        """
        gamma, beta = self.get_gamma_beta_from_text(new_text_tokens)
        new_action = self.apply_film_to_action(action, gamma, beta)
        return new_action, gamma, beta

    def loss(self, image, text_tokens, state, actions):
        """
        Compute the loss of the diffusion policy head given the image, text tokens, state, and actions.
        """
        fused_context = self.encode_obs(image, text_tokens, state)
        return self.diffusion_head.loss(actions, fused_context)

    def act(self, image, text_tokens, state):
        """
        image: (B, 3, H, W)
        text_tokens: (B, T_text)
        state: (B, state_dim)
        returns: (B, action_dim)
        """
        fused_context = self.encode_obs(image, text_tokens, state)
        actions = self.diffusion_head.sample(fused_context)
        return actions
