import torch
import torch.nn as nn
from models.audio_encoder import AudioEncoder
from models.audio_decoder import AudioDecoder
from models.tabular import TabularTransformerEncoder, TabularHead
from utils import reparameterize

class AudioCondVAE(nn.Module):
    def __init__(self, tab_spec, img_feat_dim=256, z_dim=64, d_model=128):
        super().__init__()

        self.u_enc = TabularTransformerEncoder(tab_spec, d_model=d_model)
        self.x_enc = AudioEncoder(out_dim=img_feat_dim)

        self.fuse = nn.Sequential(
            nn.Linear(img_feat_dim + d_model, 256), nn.GELU(),
            nn.Linear(256, 256), nn.GELU(),
        )

        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

        self.x_dec = AudioDecoder(z_dim=z_dim)
        self.u_head = TabularHead(z_dim=z_dim, spec=tab_spec)

    def encode(self, mel, u_cat, u_num):
        x_feat = self.x_enc(mel)
        u_feat = self.u_enc(u_cat, u_num)
        h = self.fuse(torch.cat([x_feat, u_feat], dim=1))
        return self.mu(h), self.logvar(h)

    def forward(self, mel, u_cat, u_num):
        mu, logvar = self.encode(mel, u_cat, u_num)
        z = reparameterize(mu, logvar)

        mel_hat = self.x_dec(z)
        cat_logits, num_pred = self.u_head(z)

        return {
            "mu": mu,
            "logvar": logvar,
            "z": z,
            "mel_hat": mel_hat,
            "cat_logits": cat_logits,
            "num_pred": num_pred
        }
