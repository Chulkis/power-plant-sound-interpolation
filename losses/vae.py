import torch
import torch.nn.functional as F

from config import BETA

def kl_divergence(mu, logvar):
    return 0.5 * torch.mean(torch.sum(
        torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=1))

def vae_loss(mel, mel_hat, mu, logvar, num_pred, u_num,
             cat_logits, u_cat, beta=BETA, return_dict=False):

    recon = F.l1_loss(mel_hat, mel)
    kl = kl_divergence(mu, logvar)

    voltage_loss = F.mse_loss(num_pred["voltage"], u_num["voltage"])

    toy_loss = F.cross_entropy(cat_logits["toy_id"], u_cat["toy_id"])
    total = recon + beta * kl + voltage_loss + toy_loss
    losses = {
    "total": total,
    "recon": recon,
    "kl": kl,
    "voltage": voltage_loss,
    "toy": toy_loss
    }

    return losses if return_dict else total
