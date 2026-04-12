from __future__ import annotations

import torch
import torch.nn as nn


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 128):
        super().__init__()
        if image_size != 128:
            raise ValueError("This baseline architecture expects 128x128 images.")

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.to_latent = nn.Linear(256 * 8 * 8, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 256 * 8 * 8)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        return self.to_latent(features.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.from_latent(latent).view(-1, 256, 8, 8)
        return self.decoder(decoded)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))
