from __future__ import annotations

from typing import Literal

import torch
from torch import nn

ActivationName = Literal["relu", "leaky_relu", "elu", "gelu"]
NormName = Literal["batchnorm", "none"]


def _make_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    if name == "elu":
        return nn.ELU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation: {name}")


def _generator_block(
    in_channels: int,
    out_channels: int,
    activation: ActivationName,
    dropout: float,
    norm: NormName,
) -> list[nn.Module]:
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
    ]

    if norm == "batchnorm":
        layers.append(nn.BatchNorm2d(out_channels))

    layers.append(_make_activation(activation))

    if dropout > 0.0:
        layers.append(nn.Dropout2d(dropout))

    return layers


class Generator(nn.Module):
    """DCGAN-style generator with configurable activation, dropout, and normalization."""

    def __init__(
        self,
        latent_dim: int = 128,
        out_channels: int = 3,
        base_channels: int = 64,
        activation: ActivationName = "leaky_relu",
        dropout: float = 0.0,
        norm: NormName = "batchnorm",
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        self.project = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, base_channels * 8, kernel_size=4, stride=1, padding=0, bias=False),
            *(
                [nn.BatchNorm2d(base_channels * 8)]
                if norm == "batchnorm"
                else []
            ),
            _make_activation(activation),
            *(
                [nn.Dropout2d(dropout)]
                if dropout > 0.0
                else []
            ),
        )

        self.net = nn.Sequential(
            *_generator_block(base_channels * 8, base_channels * 4, activation, dropout, norm),
            *_generator_block(base_channels * 4, base_channels * 2, activation, dropout, norm),
            *_generator_block(base_channels * 2, base_channels, activation, dropout, norm),
            nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim == 2:
            z = z.unsqueeze(-1).unsqueeze(-1)

        if z.ndim != 4:
            raise ValueError("Input noise z must have shape (N, latent_dim) or (N, latent_dim, 1, 1)")

        if z.size(1) != self.latent_dim:
            raise ValueError(f"Expected latent dim {self.latent_dim}, got {z.size(1)}")

        x = self.project(z)
        return self.net(x)
