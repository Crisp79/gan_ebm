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

            # --- Update energy network k times ---
            for _ in range(k):
                z = torch.randn(batch_size, latent_dim, 1, 1, device=device)
                fake_imgs = generator(z)

                e_real = energy_net(real_imgs).mean()
                e_fake = energy_net(fake_imgs.detach()).mean()

                gp = gradient_penalty(energy_net, real_imgs, fake_imgs.detach(), device, gp_lambda)

                loss_E = e_real - e_fake + gp

                opt_E.zero_grad()
                loss_E.backward()
                opt_E.step()

            # --- Update generator once ---
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
        activation=config.get("activation", "relu"),
    ).to(device)

    E = EnergyDiscriminator(
        channels=config.get("d_channels", [64, 128, 256, 512]),
        use_batchnorm=config.get("use_batchnorm", True),
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
        gp_lambda=config.get("gp_lambda", 10.0),
    )

    return G, E, G_losses, E_losses


if __name__ == "__main__":
    print("train_gan_ebm.py is a module that provides EBM GAN training functions.")
