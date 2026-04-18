import torch
import torch.nn as nn
import torch.nn.functional as F

class EnergyDiscriminator(nn.Module):
    """Deep Energy Model that returns a scalar energy per sample.
    
    Implements Equation 11 from "Deep Directed Generative Models 
    with Energy-Based Probability Estimation":
    E(x) = (1/sigma^2)*x^T*x - b^T*x - sum(log(1 + exp(W_i^T*f(x) + b_i)))
    """

    def __init__(
        self,
        channels=[128, 256, 512, 1024],
        in_channels=3,
        img_size=64,
        num_experts=1024,
        use_batchnorm=True,
        activation="leakyrelu",
        dropout=0.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        
        # Parameters for the raw input energy terms: (1/sigma^2)*x^T*x and b^T*x
        self.sigma_sq = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.zeros(in_channels * img_size * img_size))

        layers = []

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

        # First conv (no batchnorm) - Kernel size 5, Stride 2, Padding 2
        layers.append(nn.Conv2d(in_channels, channels[0], kernel_size=5, stride=2, padding=2))
        layers.append(get_activation())
        if dropout and dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        in_ch = channels[0]

        # Remaining convolutional layers
        for ch in channels[1:]:
            layers.append(nn.Conv2d(in_ch, ch, kernel_size=5, stride=2, padding=2))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(ch))
            layers.append(get_activation())
            if dropout and dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            in_ch = ch

        # Feature extractor f_varphi(x)
        self.features = nn.Sequential(*layers)

        # Calculate spatial dimensions after convolutions (4 layers of stride 2 -> img_size / 16)
        out_spatial = img_size // (2 ** len(channels))
        
        self.flatten = nn.Flatten()
        
        # Linear projection to create the 'experts'
        self.fc_experts = nn.Linear(in_ch * out_spatial * out_spatial, num_experts)
        self.fc_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x):
        """Return scalar energy values of shape (N,)."""
        
        # 1. Calculate energy terms operating directly on the raw input x
        x_flat = x.view(x.size(0), -1)
        
        # (1/sigma^2) * x^T * x
        x_sq_sum = torch.sum(x_flat ** 2, dim=1)
        term1 = (1.0 / self.sigma_sq) * x_sq_sum
        
        # b^T * x
        term2 = torch.matmul(x_flat, self.b)
        
        # 2. Calculate the energy term from the deep feature experts
        f = self.features(x)
        f = self.flatten(f)
        f = self.fc_dropout(f)
        
        # Calculate W_i^T * f_varphi(x) + b_i
        expert_activations = self.fc_experts(f)
        
        # The sum of softplus represents: sum(log(1 + exp(activations)))
        term3 = torch.sum(F.softplus(expert_activations), dim=1)
        
        # 3. Combine to form the final Product of Experts energy equation
        energy = term1 - term2 - term3
        
        return energy

    def features_before_fc(self, x):
        """Return feature map before the final linear layer."""
        return self.features(x)
