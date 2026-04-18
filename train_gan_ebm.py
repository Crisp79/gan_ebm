from tqdm.notebook import tqdm
import torch
import torch.nn as nn
from torch import autograd

from generator import Generator
from energy_discriminator import EnergyDiscriminator


def gradient_penalty(energy_net, real, fake, device, gp_lambda=10.0):
    batch_size = real.size(0)
    eps = torch.rand(batch_size, 1, 1, 1, device=device)
    x_hat = eps * real + (1 - eps) * fake
    x_hat.requires_grad_(True)

    e_hat = energy_net(x_hat)
    # sum energies to get scalar for grad
    grads = autograd.grad(
        outputs=e_hat.sum(), inputs=x_hat, create_graph=True
    )[0]

    grads = grads.view(batch_size, -1)
    grad_norm = grads.norm(2, dim=1)
    gp = gp_lambda * ((grad_norm - 1) ** 2).mean()
    return gp


def build_gan_models(config, device):
    """Create generator and energy-net pair from config."""
    generator = Generator(
        latent_dim=config.get("latent_dim", 128),
        channels=config.get("g_channels", [512, 256, 128, 64]),
        use_batchnorm=config.get("use_batchnorm", True),
        activation=config.get("g_activation", config.get("activation", "relu")),
        dropout=config.get("g_dropout", config.get("dropout", 0.0)),
    ).to(device)

    energy_net = EnergyDiscriminator(
        channels=config.get("d_channels", [64, 128, 256, 512]),
        use_batchnorm=config.get("use_batchnorm", True),
        activation=config.get("e_activation", "leakyrelu"),
        dropout=config.get("e_dropout", config.get("dropout", 0.0)),
    ).to(device)

    return generator, energy_net


def train_gan_with_epoch_callback(
    generator,
    energy_net=None,
    dataloader=None,
    device=None,
    epochs=20,
    latent_dim=128,
    lr=2e-4,
    e_steps=1,
    g_steps=1,
    real_ratio=0.5,
    gp_lambda=10.0,
    epoch_callback=None,
    discriminator=None,
):
    """Train EBM-style GAN and invoke callback after each epoch with latest losses.

    This is an EBM-adapted version of a standard GAN training loop where the
    discriminator is replaced by an energy network (`energy_net`). The energy
    network is trained to assign lower energy to real images and higher energy
    to generated images; the generator is trained to produce images with low
    energy.
    """
    if energy_net is None:
        energy_net = discriminator
    if energy_net is None:
        raise ValueError("Either `energy_net` or `discriminator` must be provided.")
    if dataloader is None:
        raise ValueError("`dataloader` must be provided.")
    if device is None:
        raise ValueError("`device` must be provided.")

    opt_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_E = torch.optim.Adam(energy_net.parameters(), lr=lr, betas=(0.5, 0.999))

    G_losses = []
    E_losses = []

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for real_imgs, _ in loop:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)

            # --- Update energy network e_steps times ---
            n_real = max(1, min(batch_size, int(batch_size * real_ratio)))
            n_fake = max(1, batch_size - n_real)

            for _ in range(e_steps):
                idx = torch.randperm(batch_size, device=device)[:n_real]
                real_subset = real_imgs[idx]

                z = torch.randn(n_fake, latent_dim, 1, 1, device=device)
                fake_imgs = generator(z)

                e_real = energy_net(real_subset).mean()
                e_fake = energy_net(fake_imgs.detach()).mean()

                gp = gradient_penalty(energy_net, real_subset, fake_imgs.detach(), device, gp_lambda)

                loss_E = e_real - e_fake + gp

                opt_E.zero_grad()
                loss_E.backward()
                opt_E.step()

            # --- Update generator g_steps times ---
            for _ in range(g_steps):
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_imgs = generator(z)
                loss_G = energy_net(fake_imgs).mean()

                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

            loop.set_postfix({
                "E_loss": f"{loss_E.item():.4f}",
                "G_loss": f"{loss_G.item():.4f}",
            })

        g_loss_epoch = float(loss_G.item())
        e_loss_epoch = float(loss_E.item())
        G_losses.append(g_loss_epoch)
        E_losses.append(e_loss_epoch)

        if epoch_callback is not None:
            try:
                epoch_callback(
                    epoch=epoch,
                    generator=generator,
                    energy_net=energy_net,
                    g_loss=g_loss_epoch,
                    e_loss=e_loss_epoch,
                )
            except TypeError:
                # Backward compatibility with old discriminator callback signatures.
                epoch_callback(
                    epoch=epoch,
                    generator=generator,
                    discriminator=energy_net,
                    g_loss=g_loss_epoch,
                    d_loss=e_loss_epoch,
                )

    return G_losses, E_losses


def train_gan_ebm(
    generator,
    energy_net,
    dataloader,
    device,
    epochs=20,
    latent_dim=128,
    lr_G=2e-4,
    lr_E=1e-4,
    k=5,
    g_steps=1,
    real_ratio=0.5,
    gp_lambda=10.0,
    epoch_callback=None,
    checkpoint_callback=None,
):
    """Train a generator and an energy-based discriminator.

    Args:
        generator: nn.Module that maps z -> image
        energy_net: nn.Module that maps image -> scalar energy
        dataloader: iterable yielding (images, labels)
        device: torch device
        epochs: number of epochs
        latent_dim: latent noise dimension
        lr_G, lr_E: learning rates for generator and energy net
        k: number of energy updates per generator update
        gp_lambda: gradient penalty weight
    """
    opt_G = torch.optim.Adam(generator.parameters(), lr=lr_G, betas=(0.5, 0.999))
    opt_E = torch.optim.Adam(energy_net.parameters(), lr=lr_E, betas=(0.5, 0.999))

    G_losses = []
    E_losses = []

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for real_imgs, _ in loop:
            real_imgs = real_imgs.to(device)
            batch_size = real_imgs.size(0)
            n_real = max(1, min(batch_size, int(batch_size * real_ratio)))
            n_fake = max(1, batch_size - n_real)

            # --- Update energy network k times ---
            for _ in range(k):
                idx = torch.randperm(batch_size, device=device)[:n_real]
                real_subset = real_imgs[idx]

                z = torch.randn(n_fake, latent_dim, 1, 1, device=device)
                fake_imgs = generator(z)

                e_real = energy_net(real_subset).mean()
                e_fake = energy_net(fake_imgs.detach()).mean()

                gp = gradient_penalty(energy_net, real_subset, fake_imgs.detach(), device, gp_lambda)

                loss_E = e_real - e_fake + gp

                opt_E.zero_grad()
                loss_E.backward()
                opt_E.step()

            # --- Update generator g_steps times ---
            for _ in range(g_steps):
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_imgs = generator(z)
                loss_G = energy_net(fake_imgs).mean()

                opt_G.zero_grad()
                loss_G.backward()
                opt_G.step()

            loop.set_postfix({
                "E_loss": f"{loss_E.item():.4f}",
                "G_loss": f"{loss_G.item():.4f}",
            })

        g_loss_epoch = float(loss_G.item())
        e_loss_epoch = float(loss_E.item())

        G_losses.append(g_loss_epoch)
        E_losses.append(e_loss_epoch)

        if epoch_callback is not None:
            epoch_callback(
                epoch=epoch,
                generator=generator,
                energy_net=energy_net,
                g_loss=g_loss_epoch,
                e_loss=e_loss_epoch,
            )

        if checkpoint_callback is not None:
            checkpoint_callback(epoch)

    return G_losses, E_losses


def train_gan_ebm_full(config, device, train_loader, epochs=10):
    G = Generator(
        latent_dim=config.get("latent_dim", 128),
        channels=config.get("g_channels", [512, 256, 128, 64]),
        use_batchnorm=config.get("use_batchnorm", True),
        activation=config.get("g_activation", config.get("activation", "relu")),
        dropout=config.get("g_dropout", config.get("dropout", 0.0)),
    ).to(device)

    E = EnergyDiscriminator(
        channels=config.get("d_channels", [64, 128, 256, 512]),
        use_batchnorm=config.get("use_batchnorm", True),
        activation=config.get("e_activation", "leakyrelu"),
        dropout=config.get("e_dropout", config.get("dropout", 0.0)),
    ).to(device)

    G_losses, E_losses = train_gan_ebm(
        generator=G,
        energy_net=E,
        dataloader=train_loader,
        device=device,
        epochs=epochs,
        latent_dim=config.get("latent_dim", 128),
        lr_G=config.get("lr_G", config.get("lr", 2e-4)),
        lr_E=config.get("lr_E", 1e-4),
        k=config.get("k_steps", 5),
        g_steps=config.get("g_steps", 1),
        real_ratio=config.get("real_ratio", 0.5),
        gp_lambda=config.get("gp_lambda", 10.0),
    )

    return G, E, G_losses, E_losses


if __name__ == "__main__":
    print("train_gan_ebm.py is a module that provides EBM GAN training functions.")
