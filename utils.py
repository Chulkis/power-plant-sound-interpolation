import torch
import math
import torch.nn as nn
import torchaudio
import os

from config import *

def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # z = mu + eps * sigma
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1), :]
    

mel_tf = torchaudio.transforms.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
)

amp_to_db = torchaudio.transforms.AmplitudeToDB()

def wav_to_mel(wav):
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    mel = mel_tf(wav)
    mel = amp_to_db(mel)
    return mel


inv_mel = torchaudio.transforms.InverseMelScale(
    n_stft=N_FFT//2+1, n_mels=N_MELS, sample_rate=SR
)

griffin = torchaudio.transforms.GriffinLim(
    n_fft=N_FFT, hop_length=HOP
)

def mel_to_wav(mel_db):
    mel = torch.pow(10.0, mel_db / 20.0)
    spec = inv_mel(mel)
    wav = griffin(spec)
    return wav