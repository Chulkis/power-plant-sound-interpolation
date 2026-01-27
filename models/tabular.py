from dataclasses import dataclass
from typing import Dict, List, Tuple
import math

import torch
import torch.nn as nn

@dataclass
class TabularSpec:
    # Список (name, num_classes, emb_dim)
    categorical: List[Tuple[str, int, int]]
    # Список (name,) для числовых
    numerical: List[str]

class SinusoidalPositionalEmbedding(nn.Module):
    """Классическая синусоидальная позиционка для последовательности длины L."""
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        return self.pe[:, : x.size(1), :]

class TabularTransformerEncoder(nn.Module):
    """
    Делает токены:
    - один токен на каждый categorical признак (Embedding)
    - один токен на каждый numerical признак (Linear projection из 1 -> d_model)
    Затем TransformerEncoder -> агрегируем (mean pooling) -> u_repr.
    """
    def __init__(
        self,
        spec: TabularSpec,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 128,
    ):
        super().__init__()
        self.spec = spec
        self.d_model = d_model

        self.cat_embeddings = nn.ModuleDict()
        self.cat_to_d = nn.ModuleDict()
        for name, n_classes, emb_dim in spec.categorical:
            self.cat_embeddings[name] = nn.Embedding(n_classes, emb_dim)
            self.cat_to_d[name] = nn.Linear(emb_dim, d_model)

        self.num_to_d = nn.ModuleDict({name: nn.Linear(1, d_model) for name in spec.numerical})

        self.pos_emb = SinusoidalPositionalEmbedding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u_cat: Dict[str, torch.Tensor], u_num: Dict[str, torch.Tensor]) -> torch.Tensor:
        tokens = []

        for name, _, _ in self.spec.categorical:
            x = self.cat_embeddings[name](u_cat[name])
            x = self.cat_to_d[name](x)
            tokens.append(x.unsqueeze(1))

        for name in self.spec.numerical:
            x = u_num[name]
            if x.dim() == 1:
                x = x.unsqueeze(1)
            x = self.num_to_d[name](x)
            tokens.append(x.unsqueeze(1))

        x = torch.cat(tokens, dim=1) 
        x = x + self.pos_emb(x)
        x = self.encoder(x)
        x = self.norm(x)
        u_repr = x.mean(dim=1) 
        return u_repr

class TabularHead(nn.Module):
    def __init__(self, z_dim: int, spec: TabularSpec, hidden: int = 256):
        super().__init__()
        self.spec = spec
        self.backbone = nn.Sequential(
            nn.Linear(z_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.cat_out = nn.ModuleDict()
        for name, n_classes, _ in spec.categorical:
            self.cat_out[name] = nn.Linear(hidden, n_classes)
        self.num_out = nn.ModuleDict({name: nn.Linear(hidden, 1) for name in spec.numerical})
    def forward(self, z: torch.Tensor):
        h = self.backbone(z)
        cat_logits = {name: head(h) for name, head in self.cat_out.items()}
        num_pred = {name: head(h).squeeze(1) for name, head in self.num_out.items()}
        return cat_logits, num_pred

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std