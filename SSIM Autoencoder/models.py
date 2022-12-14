import torch
import torch.nn as nn
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self, n_channels):
        super().__init__()        

        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels, 32, 4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 100, 8, stride=1, padding=0),
            nn.LeakyReLU(0.2),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 32, 8, stride=1, padding=0), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 64, 3, stride=1, padding=1), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 128, 3, stride=1, padding=1), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), 
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 3, stride=1, padding=1),  
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),  
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, n_channels, 4, stride=2, padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

