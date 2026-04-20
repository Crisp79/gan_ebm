from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn.utils import spectral_norm

ActivationName = Literal["relu", "leaky_relu", "elu", "gelu"]
NormName = Literal["spectral", "none"]


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


def _maybe_sn(layer: nn.Module, norm: NormName) -> nn.Module:
    return spectral_norm(layer) if norm == "spectral" else layer


def _sample_independent_norm(num_channels: int) -> nn.Module:
    return nn.GroupNorm(1, num_channels)


class Encoder(nn.Module):
    """Image encoder with a strict linear bottleneck."""

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        embedding_channels: int,
        activation: ActivationName,
        dropout: float,
        norm: NormName,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            _maybe_sn(nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1, bias=True), norm),
            _sample_independent_norm(base_channels),
            _make_activation(activation),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            _maybe_sn(
                nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=True),
                norm,
            ),
            _sample_independent_norm(base_channels * 2),
            _make_activation(activation),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            _maybe_sn(
                nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=True),
                norm,
            ),
            _sample_independent_norm(base_channels * 4),
            _make_activation(activation),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            _maybe_sn(
                nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=4, stride=2, padding=1, bias=True),
                norm,
            ),
            _sample_independent_norm(base_channels * 8),
            _make_activation(activation),
        )

        self.flatten = nn.Flatten()
        self.project = _maybe_sn(nn.Linear(base_channels * 8 * 4 * 4, embedding_channels, bias=True), norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.net(x)
        flat = self.flatten(features)
        return self.project(flat)


class Decoder(nn.Module):
    """Decoder that reconstructs an image from the linear bottleneck."""

    def __init__(
        self,
        out_channels: int,
        base_channels: int,
        embedding_channels: int,
        activation: ActivationName,
        dropout: float,
        norm: NormName,
    ) -> None:
        super().__init__()

        self.expand = _maybe_sn(nn.Linear(embedding_channels, base_channels * 8 * 4 * 4, bias=True), norm)

        self.net = nn.Sequential(
            _make_activation(activation),
            nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
            _maybe_sn(
                nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=4, stride=2, padding=1, bias=True),
                norm,
            ),
            _sample_independent_norm(base_channels * 4),
            _make_activation(activation),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            _maybe_sn(
                nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=4, stride=2, padding=1, bias=True),
                norm,
            ),
            _sample_independent_norm(base_channels * 2),
            _make_activation(activation),
            nn.Dropout2d(dropout) if dropout > 0.0 else nn.Identity(),
            _maybe_sn(nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=4, stride=2, padding=1, bias=True), norm),
            _sample_independent_norm(base_channels),
            _make_activation(activation),
            _maybe_sn(nn.ConvTranspose2d(base_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True), norm),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 2:
            raise ValueError("Decoder expects bottleneck shape (N, embedding_dim)")

        x = self.expand(z)
        x = x.view(z.size(0), -1, 4, 4)
        return self.net(x)


class EnergyModel(nn.Module):
    """EBGAN autoencoder discriminator that returns reconstruction and bottleneck."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        embedding_channels: int = 128,
        activation: ActivationName = "leaky_relu",
        dropout: float = 0.0,
        norm: NormName = "spectral",
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            embedding_channels=embedding_channels,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )
        self.decoder = Decoder(
            out_channels=in_channels,
            base_channels=base_channels,
            embedding_channels=embedding_channels,
            activation=activation,
            dropout=dropout,
            norm=norm,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("EnergyModel expects image tensor with shape (N, C, H, W)")

        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction, bottleneck

    def energy(self, x: torch.Tensor, reconstruction: torch.Tensor | None = None) -> torch.Tensor:
        if reconstruction is None:
            reconstruction, _ = self.forward(x)

        return (reconstruction - x).pow(2).flatten(start_dim=1).mean(dim=1)


# Backward-compatible alias for the training code and earlier references.
Discriminator = EnergyModel
