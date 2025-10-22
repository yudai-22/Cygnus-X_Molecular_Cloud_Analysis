import torch.nn as nn


class Conv3dAutoencoder(nn.Module):
    def __init__(self, latent):
        super(Conv3dAutoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv3d(64, 32, kernel_size=3, stride=2, padding=0),nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32 * 14 * 13 * 13, latent),  # Adjust the size based on the flattened output
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent, 32 * 14 * 13 * 13),
            nn.Unflatten(1, (32, 14, 13, 13)),# Unflatten back
            nn.ConvTranspose3d(32, 32, kernel_size=5, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose3d(32, 32, kernel_size=5, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),  # to scale output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x