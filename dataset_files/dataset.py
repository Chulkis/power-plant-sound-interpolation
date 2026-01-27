from pathlib import Path
import random
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from config import SR, SEGMENT_LEN, SPEED_TO_VOLTAGE, TOY_MAP
from augment import motor_augment
from utils import wav_to_mel

class ToyADMOSDataset(Dataset):
    def __init__(self, root, augment=False):
        self.root = Path(root)
        self.augment = augment
        self.files = []

        self.toy_map = TOY_MAP

        for toy_dir in self.root.iterdir():
            if not toy_dir.is_dir():
                continue

            toy_name = toy_dir.name
            toy_id = self.toy_map[toy_name]

            for speed_dir in toy_dir.iterdir():
                if not speed_dir.is_dir():
                    continue

                if not speed_dir.name.startswith("speed_"):
                    continue

                speed = int(speed_dir.name.replace("speed_", ""))

                for wav_path in speed_dir.glob("*.wav"):
                    self.files.append({
                        "path": wav_path,
                        "toy_id": toy_id,
                        "toy_name": toy_name,
                        "speed": speed
                    })

        print(f"[ToyADMOSDataset] Found {len(self.files)} files")
        print("Toys:", self.toy_map)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        item = self.files[idx]

        wav, sr = torchaudio.load(item["path"])
        wav = torchaudio.functional.resample(wav, sr, SR)[0]

        # crop / pad
        if wav.size(0) > SEGMENT_LEN:
            i = random.randint(0, wav.size(0) - SEGMENT_LEN)
            wav = wav[i:i + SEGMENT_LEN]
        else:
            wav = F.pad(wav, (0, SEGMENT_LEN - wav.size(0)))

        # augment
        if self.augment:
            wav = motor_augment(wav)

        # wav -> mel
        mel = wav_to_mel(wav)
        if mel.size(-1) < 96:
            mel = F.pad(mel, (0, 96 - mel.size(-1)))
        else:
            mel = mel[:, :, :96]
        mel = torch.log(mel + 1e-5)

        speed = item["speed"]
        voltage = SPEED_TO_VOLTAGE[speed]

        return {
            "mel": mel,
            "u_cat": {
                "toy_id": torch.tensor(item["toy_id"], dtype=torch.long)
            },
            "u_num": {
                "voltage": torch.tensor(voltage, dtype=torch.float32),
            }
        }
