SR = 48000
SEGMENT_LEN = 48000

N_MELS = 128
N_FFT = 1024
HOP = 512

DATA_ROOT = "data/dataset/ToyCar"
BATCH_SIZE = 16
LR = 2e-4
EPOCHS = 20
SAVE_EVERY = 20
INTERP_EVERY = 10

NUM_WORKERS = 4

Z_DIM = 64
IMG_FEAT_DIM = 256
BETA = 0.5

SPEED_TO_VOLTAGE = {
    1: 2.8,
    2: 3.1,
    3: 3.4,
    4: 3.7,
    5: 4.0
}

TOY_MAP = {
    "A1": 0,
    "A2": 1,
    "B1": 2,
    "B2": 3
}
