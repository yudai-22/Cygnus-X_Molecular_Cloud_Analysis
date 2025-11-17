import torch.nn as nn


class Binary_classification(nn.Module):
    def __init__(self, latent):
        super(Binary_classification, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32), 
            nn.ReLU(True),
            nn.Conv3d(32, 32, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm3d(32), 
            nn.ReLU(True),
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64), 
            nn.ReLU(True),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=0), 
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(32 * 5 * 23 * 23, latent),  # Adjust the size based on the flattened output
            nn.Linear(latent, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x
