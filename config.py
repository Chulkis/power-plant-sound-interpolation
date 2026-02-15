SR = 48000
SEGMENT_LEN = 48000

N_MELS = 64
N_FFT = 1024
HOP = 1024
TARGET_T = 96

DATA_ROOT = "data/dataset/ToyADMOS_mel"
BATCH_SIZE = 16
LR = 2e-4
EPOCHS = 100
SAVE_EVERY = 20
INTERP_EVERY = 1 # Каждые сколько эпох интерполировать

EPOCH_LINEAR_DOWN = 20 # Эпоха, после которой линейно увеличивается BETA
EPOCH_LINEAR_UP = 50 # Эпоха, до которой линейно увеличивается BETA

NUM_WORKERS = 4

Z_DIM = 64
IMG_FEAT_DIM = 256
BETA_DOWN = 0.0
BETA_AVG = 0.1
BETA_UP = 0.5

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
