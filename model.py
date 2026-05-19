from __future__ import annotations

import torch
import torch.nn as nn


def _conv_block(in_channels: int, out_channels: int, use_batch_norm: bool) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return layers


def _deconv_block(in_channels: int, out_channels: int, use_batch_norm: bool) -> list[nn.Module]:
    layers: list[nn.Module] = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)]
    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    layers.append(nn.ReLU(inplace=True))
    return layers


def _build_encoder(use_batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_channels, out_channels in [(1, 32), (32, 64), (64, 128), (128, 256)]:
        layers.extend(_conv_block(in_channels, out_channels, use_batch_norm))
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def _build_decoder(use_batch_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for in_channels, out_channels in [(256, 128), (128, 64), (64, 32), (32, 16)]:
        layers.extend(_deconv_block(in_channels, out_channels, use_batch_norm))
    layers.extend(
        [
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        ]
    )
    return nn.Sequential(*layers)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 128, use_batch_norm: bool = False):
        super().__init__()
        if image_size != 128:
            raise ValueError("This baseline architecture expects 128x128 images.")

        self.model_type = "ae"
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.encoder = _build_encoder(use_batch_norm=use_batch_norm)
        self.to_latent = nn.Linear(256 * 8 * 8, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = _build_decoder(use_batch_norm=use_batch_norm)

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.encoder(inputs)
        return self.to_latent(features.flatten(start_dim=1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.from_latent(latent).view(-1, 256, 8, 8)
        return self.decoder(decoded)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(inputs))


class ConvVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 128, image_size: int = 128, use_batch_norm: bool = False):
        super().__init__()
        if image_size != 128:
            raise ValueError("This VAE architecture expects 128x128 images.")

        self.model_type = "vae"
        self.latent_dim = latent_dim
        self.use_batch_norm = use_batch_norm
        self.encoder = _build_encoder(use_batch_norm=use_batch_norm)
        self.to_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.to_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        self.from_latent = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = _build_decoder(use_batch_norm=use_batch_norm)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(inputs).flatten(start_dim=1)
        return self.to_mu(features), self.to_logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        decoded = self.from_latent(latent).view(-1, 256, 8, 8)
        return self.decoder(decoded)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(inputs)
        latent = self.reparameterize(mu, logvar)
        return {
            "reconstruction": self.decode(latent),
            "mu": mu,
            "logvar": logvar,
        }


def build_model(model_type: str, latent_dim: int, image_size: int, use_batch_norm: bool = False) -> nn.Module:
    if model_type == "ae":
        return ConvAutoencoder(latent_dim=latent_dim, image_size=image_size, use_batch_norm=use_batch_norm)
    if model_type == "vae":
        return ConvVariationalAutoencoder(latent_dim=latent_dim, image_size=image_size, use_batch_norm=use_batch_norm)
    raise ValueError(f"Unsupported model_type: {model_type}")
