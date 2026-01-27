import torch
import torch.nn.functional as F
import random

# Аугментации для wav

def add_noise(wav, min_snr=25, max_snr=45):
    snr = torch.empty(1).uniform_(min_snr, max_snr)
    sig_power = wav.pow(2).mean()
    noise_power = sig_power / (10 ** (snr / 10))
    noise = torch.randn_like(wav) * torch.sqrt(noise_power)
    return wav + noise


def random_gain(wav, min_gain=0.8, max_gain=1.2):
    gain = torch.empty(1).uniform_(min_gain, max_gain)
    return wav * gain


def random_time_shift(wav, max_shift=400):
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(wav, shifts=shift, dims=0)


def random_lowpass(wav, max_alpha=0.15):
    alpha = random.uniform(0.01, max_alpha)
    y = torch.zeros_like(wav)
    y[0] = wav[0]
    for i in range(1, len(wav)):
        y[i] = alpha * wav[i] + (1 - alpha) * y[i - 1]
    return y


def motor_augment(wav, p=0.8):
    if random.random() > p:
        return wav

    if random.random() < 0.7:
        wav = add_noise(wav)

    if random.random() < 0.5:
        wav = random_gain(wav)

    if random.random() < 0.3:
        wav = random_time_shift(wav)

    if random.random() < 0.3:
        wav = random_lowpass(wav)

    return wav

# Аугментации для mel

def freq_mask(mel, max_width=8):
    f = mel.size(1)
    w = random.randint(0, max_width)
    f0 = random.randint(0, f - w)
    mel[:, f0:f0+w, :] = 0
    return mel

def time_mask(mel, max_width=16):
    t = mel.size(2)
    w = random.randint(0, max_width)
    t0 = random.randint(0, t - w)
    mel[:, :, t0:t0+w] = 0
    return mel

def mel_augment(mel):
    if random.random() < 0.5:
        mel = mel * random.uniform(0.98, 1.02)
        mel = mel + random.uniform(-0.05, 0.05)

    if random.random() < 0.5:
        mel = mel + torch.randn_like(mel) * 0.001

    if random.random() < 0.3:
        mel = freq_mask(mel, 1)

    if random.random() < 0.3:
        mel = time_mask(mel, 2)

    return mel
