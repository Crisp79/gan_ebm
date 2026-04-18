import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        channels=[512, 256, 128, 64],
        out_channels=3,
        use_batchnorm=True,
        activation="relu"
    ):
        super().__init__()

        layers = []

        def get_activation():
            return nn.ReLU(True) if activation == "relu" else nn.LeakyReLU(0.2, True)

        # First layer (no stride)
        layers.append(nn.ConvTranspose2d(latent_dim, channels[0], 4, 1, 0))
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(channels[0]))
        layers.append(get_activation())

        # Hidden layers
        in_ch = channels[0]
        for ch in channels[1:]:
            layers.append(nn.ConvTranspose2d(in_ch, ch, 4, 2, 1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(get_activation())
            in_ch = ch

        # Output layer
        layers.append(nn.ConvTranspose2d(in_ch, out_channels, 4, 2, 1))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)