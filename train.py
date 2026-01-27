import os
import torch
from tqdm import tqdm
import random, torch, numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time

from models.cond_vae import AudioCondVAE
from dataset_files.mel_dataset import Mel_ToyADMOSDataset
from models.tabular import TabularSpec
from losses.vae import vae_loss
from config import *
from interpolate import save_interpolation
from utils import mel_to_wav

def main():
    run_name = time.strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/motor_vae_{run_name}")

    epoch_losses = {"total":0, "recon":0, "kl":0, "voltage":0, "toy":0}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    tab_spec = TabularSpec(
        categorical=[("toy_id", 4, 8)],    
        numerical=["voltage"]
    )

    ds = Mel_ToyADMOSDataset(os.path.join(DATA_ROOT, "train"), augment=True)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=True)

    model = AudioCondVAE(tab_spec)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    ref_ds = Mel_ToyADMOSDataset(os.path.join(DATA_ROOT, "train"), augment=False)

    ref_a = ref_ds[10]
    ref_b = ref_ds[200]

    ref_a = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else
            {kk: vv.unsqueeze(0).to(device) for kk, vv in v.items()}
            for k, v in ref_a.items()}

    ref_b = {k: v.unsqueeze(0).to(device) if torch.is_tensor(v) else
            {kk: vv.unsqueeze(0).to(device) for kk, vv in v.items()}
            for k, v in ref_b.items()}
    
    best_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss = 0.0

        for batch in tqdm(dl, desc=f"Epoch {epoch}/{EPOCHS}"):
            mel = batch["mel"].to(device)

            u_cat = {k: v.to(device) for k, v in batch["u_cat"].items()}
            u_num = {k: v.to(device) for k, v in batch["u_num"].items()}

            out = model(mel, u_cat, u_num)

            mel_hat = out["mel_hat"]
            losses = vae_loss(mel=mel,
                mel_hat=mel_hat,
                mu=out["mu"],
                logvar=out["logvar"],
                num_pred=out["num_pred"],
                u_num=u_num,
                cat_logits=out["cat_logits"],
                u_cat=u_cat,
                beta=BETA, return_dict=True)
            
            loss = losses["total"]

            opt.zero_grad()
            loss.backward()
            opt.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()

        for k in epoch_losses:
            epoch_losses[k] /= len(dl)

        avg_loss = epoch_losses["total"]

        print(epoch_losses)

        for name, value in epoch_losses.items():
            writer.add_scalar(f"loss/{name}", value, epoch)

        writer.add_image("mel/original", mel[0], epoch)
        writer.add_image("mel/recon", mel_hat[0], epoch)

        writer.add_histogram("latent/z", out["z"], epoch)
        writer.add_histogram("latent/mu", out["mu"], epoch)

        if epoch % SAVE_EVERY == 0:
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "loss": avg_loss
            }, f"checkpoints/epoch_{epoch}.pt")
        
        if epoch % INTERP_EVERY == 0:
            save_interpolation(
                model=model,
                a=ref_a,
                b=ref_b,
                mel_to_wav=mel_to_wav,
                out_dir=f"interps/epoch_{epoch}",
                steps=9
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "checkpoints/best.pt")

if __name__ == "__main__":
    main()
