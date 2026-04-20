import torch
import torch.nn as nn


class Generator(nn.Module):
    """Decoder-style generator mapping latent noise to image space."""

    def __init__(
        self,
        latent_dim=128,
        channels=[512, 256, 128, 64],
        out_channels=3,
        use_batchnorm_gen=False,
        activation="relu",
        dropout=0.0,
    ):
        super().__init__()

        if not channels:
            raise ValueError("`channels` must contain at least one channel size.")

        self.latent_dim = latent_dim
        layers = []

        def get_activation():
            act = str(activation).lower()
            if act == "relu":
                return nn.ReLU(True)
            if act in {"leakyrelu", "leaky_relu", "leaky"}:
                return nn.LeakyReLU(0.2, True)
            if act == "elu":
                return nn.ELU(inplace=True)
            if act == "gelu":
                return nn.GELU()
            if act in {"silu", "swish"}:
                return nn.SiLU(inplace=True)
            raise ValueError(f"Unsupported generator activation: {activation}")

        # Project z from 1x1 into a small spatial feature map.
        layers.append(nn.ConvTranspose2d(latent_dim, channels[0], 4, 1, 0))
        if use_batchnorm_gen:
            layers.append(nn.BatchNorm2d(channels[0]))
        layers.append(get_activation())
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        # Upsample progressively until final image resolution.
        in_ch = channels[0]
        for ch in channels[1:]:
            layers.append(nn.ConvTranspose2d(in_ch, ch, 4, 2, 1))
            if use_batchnorm_gen:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(get_activation())
            if dropout and dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = ch

        # Output RGB image in [-1, 1].
        layers.append(nn.ConvTranspose2d(in_ch, out_channels, 4, 2, 1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)