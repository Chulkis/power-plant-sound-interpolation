from pathlib import Path
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from config import *
from augment import mel_augment
from utils import mel_to_wav

class Mel_ToyADMOSDataset(Dataset):
    def __init__(self, mel_root, augment=False):
        self.files = list(Path(mel_root).rglob("*.pt"))
        self.augment = augment
        print("MEL ROOT:", mel_root)
        print("FOUND FILES:", len(self.files))

        print("Found mel files:", len(self.files))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]  
        item = torch.load(path)

        mel = item["mel"]
        u_cat = item["u_cat"]
        u_num = item["u_num"]

        u_cat = {k: torch.tensor(v, dtype=torch.long) for k, v in u_cat.items()}
        u_num = {k: torch.tensor(v, dtype=torch.float32) for k, v in u_num.items()}

        if mel.size(-1) > TARGET_T:
            i = random.randint(0, mel.size(-1) - TARGET_T)
            mel = mel[:, :, i:i+TARGET_T]
        else:
            mel = F.pad(mel, (0, TARGET_T - mel.size(-1)))

        if self.augment:
            mel = mel_augment(mel)


        return {
            "mel": mel,
            "u_cat": u_cat,
            "u_num": u_num
        }
