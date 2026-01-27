import torch.nn as nn
from config import N_MELS

class AudioDecoder(nn.Module):
    def __init__(self, z_dim=64, n_mels=N_MELS, time_steps=96):
        super().__init__()
        self.h0, self.w0 = N_MELS // 16, time_steps // 16
        self.fc = nn.Linear(z_dim, 256*self.h0*self.w0)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.GELU(),
            nn.ConvTranspose2d(128,64,4,2,1), nn.GELU(),
            nn.ConvTranspose2d(64,32,4,2,1), nn.GELU(),
            nn.ConvTranspose2d(32,1,4,2,1)
        )

    def forward(self, z):
        x = self.fc(z).view(z.size(0),256,self.h0,self.w0)
        return self.deconv(x)
