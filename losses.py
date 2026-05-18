from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def unpack_reconstruction(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, dict):
        return model_output["reconstruction"]
    return model_output


def ssim_per_sample(inputs: torch.Tensor, reconstructions: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    padding = window_size // 2
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = F.avg_pool2d(inputs, kernel_size=window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(reconstructions, kernel_size=window_size, stride=1, padding=padding)

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(inputs * inputs, window_size, stride=1, padding=padding) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(reconstructions * reconstructions, window_size, stride=1, padding=padding) - mu_y_sq
    sigma_xy = F.avg_pool2d(inputs * reconstructions, window_size, stride=1, padding=padding) - mu_xy

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    ssim_map = numerator / denominator.clamp_min(1e-8)
    return ssim_map.mean(dim=(1, 2, 3)).clamp(0.0, 1.0)


def reconstruction_errors(
    inputs: torch.Tensor,
    reconstructions: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    if loss_type == "l1":
        return torch.mean(torch.abs(inputs - reconstructions), dim=(1, 2, 3))
    if loss_type == "mse_ssim":
        mse = torch.mean((inputs - reconstructions) ** 2, dim=(1, 2, 3))
        structural_error = 1.0 - ssim_per_sample(inputs, reconstructions)
        return 0.8 * mse + 0.2 * structural_error
    return torch.mean((inputs - reconstructions) ** 2, dim=(1, 2, 3))


def kl_divergence_per_sample(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def compute_loss(
    inputs: torch.Tensor,
    model_output: Any,
    loss_type: str,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    reconstructions = unpack_reconstruction(model_output)
    per_sample_score = reconstruction_errors(inputs, reconstructions, loss_type)
    reconstruction_loss = per_sample_score.mean()
    loss = reconstruction_loss
    components = {"reconstruction_loss": float(reconstruction_loss.detach().cpu())}

    if loss_type == "vae_mse_kl":
        if not isinstance(model_output, dict) or "mu" not in model_output or "logvar" not in model_output:
            raise ValueError("vae_mse_kl loss requires VAE output with mu and logvar.")
        kl_loss = kl_divergence_per_sample(model_output["mu"], model_output["logvar"]).mean()
        loss = reconstruction_loss + beta * kl_loss
        components["kl_loss"] = float(kl_loss.detach().cpu())
        components["beta"] = float(beta)

    return loss, components
