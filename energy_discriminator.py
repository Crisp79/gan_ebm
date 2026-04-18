import torch
import torch.nn as nn


class EnergyDiscriminator(nn.Module):
    """Energy-based discriminator that returns a scalar energy per sample.

    Matches the constructor signature of the existing `Discriminator` so
    it can be used as a drop-in replacement in training code.
    """

    def __init__(
        self,
        channels=[64, 128, 256, 512],
        in_channels=3,
        use_batchnorm=True,
    ):
        super().__init__()

        layers = []

        # First conv (no batchnorm)
        layers.append(nn.Conv2d(in_channels, channels[0], 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        in_ch = channels[0]

        for ch in channels[1:]:
            layers.append(nn.Conv2d(in_ch, ch, 4, 2, 1))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_ch = ch

        # feature extractor (all convs)
        self.features = nn.Sequential(*layers)

        # final projection to scalar energy (no sigmoid)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_ch * 4 * 4, 1)

    def forward(self, x):
        """Return scalar energy values of shape (N,).

        Lower energy => higher model probability (by convention).
        """
        f = self.features(x)
        f = self.flatten(f)
        e = self.fc(f)
        return e.view(-1)

    def features_before_fc(self, x):
        """Return feature map before the final linear layer (useful for debugging)."""
        return self.features(x)
