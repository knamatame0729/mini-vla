"""Encoders for images, text, and states."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoderTinyCNN(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(128, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = F.relu(self.conv1(x))     # (B, 32, 32, 32)
        x = F.relu(self.conv2(x))     # (B, 64, 16, 16)
        x = F.relu(self.conv3(x))     # (B, 128, 8, 8)
        x = x.mean(dim=[2, 3])        # Global average pooling -> (B, 128)
        x = self.proj(x)              # (B, d_model)
        x = self.ln(x)                # Layer Normalization
        return x  # (B, d_model)


class TextEncoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_word=64, d_model=128, num_layers=2, num_heads=4, dim_feedforward=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_word)
        self.pos_embed = nn.Parameter(torch.randn(1, 512, d_word))  # Support up to 512 tokens
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_word,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Project to d_model
        self.proj = nn.Linear(d_word, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids, attention_mask=None):
        # token_ids: (B, T)
        x = self.embed(token_ids)  # (B, T, d_word)
        
        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Apply transformer
        if attention_mask is not None:
            x = self.transformer(x, src_key_padding_mask=attention_mask)
        else:
            x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)  # (B, d_word)
        
        # Project and normalize
        x = self.proj(x)  # (B, d_model)
        x = self.ln(x)
        return x


class TextEncoderTinyGRU(nn.Module):
    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_word)
        self.gru = nn.GRU(d_word, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        x = self.embed(token_ids)  # (B, T, d_word)
        _, h_last = self.gru(x)
        x = h_last[0]  # (B, d_model)
        x = self.ln(x)
        return x


class StateEncoderMLP(nn.Module):
    def __init__(self, state_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.net(s)
        x = self.ln(x)
        return x
