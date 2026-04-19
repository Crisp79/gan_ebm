import torch
import random
import numpy as np
import gc
from torchvision.utils import make_grid, save_image
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # for multi-GPU
    # Crucial for local determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_samples(generator, device, latent_dim=128, n_samples=16):
    generator.eval()

    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim, 1, 1).to(device)
        samples = generator(z)

    return samples


def show_images(images, title="Images"):
    grid = make_grid(images, nrow=4, normalize=True)

    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.title(title)
    plt.axis("off")
    plt.show()


def save_samples(generator, device, path, latent_dim=128):
    samples = generate_samples(generator, device, latent_dim, n_samples=12)
    save_image(samples, path, nrow=4, normalize=True)


def clear_eval_memory():
    """Clear Python/CUDA memory after metric-heavy evaluation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def compute_fid(generator, dataloader, device, latent_dim, num_batches=10):
    fid = FrechetInceptionDistance(feature=2048).to(device)

    generator.eval()

    with torch.no_grad():
        for i, (real_imgs, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            real_imgs = real_imgs.to(device)

            # Convert to [0,255] uint8
            real_imgs_uint8 = ((real_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)

            fid.update(real_imgs_uint8, real=True)

            z = torch.randn(real_imgs.size(0), latent_dim, 1, 1).to(device)
            fake_imgs = generator(z)

            fake_imgs_uint8 = ((fake_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)

            fid.update(fake_imgs_uint8, real=False)

    score = fid.compute().item()
    del fid
    clear_eval_memory()
    return score


def compute_is(generator, device, latent_dim, n_samples=500):
    inception = InceptionScore().to(device)

    generator.eval()

    with torch.no_grad():
        for _ in range(n_samples // 32):
            z = torch.randn(32, latent_dim, 1, 1).to(device)
            fake_imgs = generator(z)

            fake_imgs_uint8 = ((fake_imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)

            inception.update(fake_imgs_uint8)

    score, _ = inception.compute()
    del inception
    clear_eval_memory()
    return score.item()
