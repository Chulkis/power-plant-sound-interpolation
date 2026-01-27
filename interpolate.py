import torch
import os
import torchaudio

from config import *

@torch.no_grad()
def save_interpolation(model, a, b, mel_to_wav, out_dir, steps=9):
    os.makedirs(out_dir, exist_ok=True)
    was_training = model.training
    model.eval()

    mu1, _ = model.encode(a["mel"], a["u_cat"], a["u_num"])
    mu2, _ = model.encode(b["mel"], b["u_cat"], b["u_num"])

    for i, t in enumerate(torch.linspace(0, 1, steps)):
        z = (1 - t) * mu1 + t * mu2
        mel_hat = model.x_dec(z)
        wav = mel_to_wav(mel_hat[0].cpu())

        path = os.path.join(out_dir, f"interp_{i:02d}.wav")
        torchaudio.save(path, wav.unsqueeze(0), SR)

    if was_training:
        model.train()

