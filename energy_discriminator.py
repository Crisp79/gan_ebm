import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class EnergyDiscriminator(nn.Module):
    """Autoencoder energy model that returns scalar reconstruction energy.

    Energy is computed per sample as the mean squared reconstruction error:
        E(x) = mean((x - recon(x))^2)
    """

    def __init__(
        self,
        channels=[128, 256, 512, 1024],
        in_channels=3,
        img_size=64,
        num_experts=1024,
        use_spectral_norm=False,
        activation="leakyrelu",
        dropout=0.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.bottleneck_dim = num_experts

        if not channels:
            raise ValueError("`channels` must contain at least one channel size.")

        def get_activation():
            act = str(activation).lower()
            if act == "relu":
                return nn.ReLU(inplace=True)
            if act in {"leakyrelu", "leaky_relu", "leaky"}:
                return nn.LeakyReLU(0.2, inplace=True)
            if act == "elu":
                return nn.ELU(inplace=True)
            if act == "gelu":
                return nn.GELU()
            if act in {"silu", "swish"}:
                return nn.SiLU(inplace=True)
            raise ValueError(f"Unsupported energy activation: {activation}")

        def maybe_apply_spectral_norm(layer):
            """Apply spectral norm if enabled."""
            if use_spectral_norm:
                return spectral_norm(layer)
            return layer

        encoder_layers = []
        in_ch = in_channels
        for ch in channels:
            encoder_layers.append(
                maybe_apply_spectral_norm(
                    nn.Conv2d(in_ch, ch, kernel_size=5, stride=2, padding=2)
                )
            )
            encoder_layers.append(get_activation())
            if dropout and dropout > 0:
                encoder_layers.append(nn.Dropout2d(dropout))
            in_ch = ch
        self.encoder = nn.Sequential(*encoder_layers)

        out_spatial = img_size // (2 ** len(channels))
        if out_spatial < 1:
            raise ValueError("`img_size` is too small for the number of downsampling layers.")

        self.feature_channels = in_ch
        self.feature_spatial = out_spatial
        self.flat_dim = in_ch * out_spatial * out_spatial

        self.to_bottleneck = maybe_apply_spectral_norm(
            nn.Linear(self.flat_dim, self.bottleneck_dim)
        )
        self.from_bottleneck = maybe_apply_spectral_norm(
            nn.Linear(self.bottleneck_dim, self.flat_dim)
        )

        decoder_layers = []
        decoder_channels = list(channels[::-1])
        for idx, ch in enumerate(decoder_channels[1:]):
            decoder_layers.append(
                maybe_apply_spectral_norm(
                    nn.ConvTranspose2d(
                        decoder_channels[idx],
                        ch,
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                    )
                )
            )
            decoder_layers.append(get_activation())
            if dropout and dropout > 0:
                decoder_layers.append(nn.Dropout2d(dropout))

        decoder_layers.append(
            maybe_apply_spectral_norm(
                nn.ConvTranspose2d(
                    decoder_channels[-1],
                    in_channels,
                    kernel_size=5,
                    stride=2,
                    padding=2,
                    output_padding=1,
                )
            )
        )
        decoder_layers.append(nn.Tanh())
        self.decoder = nn.Sequential(*decoder_layers)

        self.bottleneck_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        """Return scalar energy values of shape (N,)."""
        recon = self.reconstruct(x)
        return (x - recon).pow(2).flatten(1).mean(dim=1)

    def features_before_fc(self, x):
        """Return encoder bottleneck representation used for metric calculations."""
        return self.encode(x)

    def encode(self, x):
        """Return bottleneck representation S from the encoder."""
        features = self.encoder(x)
        flat = features.flatten(1)
        bottleneck = self.to_bottleneck(flat)
        bottleneck = self.bottleneck_dropout(bottleneck)
        return bottleneck

    def decode(self, bottleneck):
        """Decode bottleneck vector back to image space."""
        flat = self.from_bottleneck(bottleneck)
        features = flat.view(
            bottleneck.size(0),
            self.feature_channels,
            self.feature_spatial,
            self.feature_spatial,
        )
        return self.decoder(features)

    def reconstruct(self, x):
        """Autoencoder reconstruction used for energy computation."""
        return self.decode(self.encode(x))
