from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
from types import SimpleNamespace
import gc

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from typing import Optional
from torchmetrics.image.fid import FrechetInceptionDistance

from dgm import Generator
from dem import Discriminator


def _to_device(batch: Any, device: torch.device) -> torch.Tensor:
    images = batch[0] if isinstance(batch, (tuple, list)) else batch
    return images.to(device)


def _clear_runtime_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def clear_model_from_memory(*models: nn.Module) -> None:
    for model in models:
        del model
    _clear_runtime_memory()


def _split_real_fake_counts(batch_size: int, real_ratio: float) -> tuple[int, int]:
    real_count = int(batch_size * real_ratio)
    fake_count = batch_size - real_count
    return real_count, fake_count


def _pulling_away_term(embedding: torch.Tensor) -> torch.Tensor:
    """Compute pulling-away term on flattened bottleneck features."""
    n = embedding.size(0)
    if n < 2:
        return embedding.new_tensor(0.0)

    flat = embedding.flatten(start_dim=1)
    normed = F.normalize(flat, p=2, dim=1)
    similarity = normed @ normed.t()

    # Exclude diagonal and average over distinct pairs.
    off_diag_sum = (similarity.pow(2).sum() - torch.diagonal(similarity.pow(2)).sum())
    return off_diag_sum / (n * (n - 1))


def _make_fixed_noise(sample_count: int, latent_dim: int, seed: int, device: torch.device) -> torch.Tensor:
    g = torch.Generator(device=device.type)
    g.manual_seed(seed)
    return torch.randn(sample_count, latent_dim, 1, 1, generator=g, device=device)


def estimate_initial_energy(
    discriminator: Discriminator,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 1,
) -> float:
    """Estimate the initial mean reconstruction energy on real data for margin calibration."""
    discriminator.eval()

    energies: list[float] = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            real = _to_device(batch, device)
            reconstruction, _ = discriminator(real)
            energy = discriminator.energy(real, reconstruction).mean()
            energies.append(float(energy.item()))

            del real, reconstruction, energy

    _clear_runtime_memory()
    return sum(energies) / max(len(energies), 1)


def _save_sample_grid(
    generator: Generator,
    fixed_noise: torch.Tensor,
    sample_path: Path,
    cleanup_after: bool,
) -> None:
    generator.eval()
    with torch.no_grad():
        images = generator(fixed_noise)
        grid = make_grid(images, nrow=4, normalize=True)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(grid, sample_path)

    if cleanup_after:
        del images
        del grid
        _clear_runtime_memory()


def evaluate_model(
    generator: Generator,
    discriminator: Discriminator,
    dataloader: DataLoader,
    device: torch.device,
    latent_dim: int,
    max_batches: int = 10,
) -> dict[str, float]:
    generator.eval()
    discriminator.eval()

    e_real_sum = 0.0
    e_fake_sum = 0.0
    pt_sum = 0.0
    n_batches = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break

            real = _to_device(batch, device)
            bs = real.size(0)

            z = torch.randn(bs, latent_dim, 1, 1, device=device)
            fake = generator(z)

            real_recon, _ = discriminator(real)
            fake_recon, fake_embed = discriminator(fake)

            e_real = discriminator.energy(real, real_recon).mean()
            e_fake = discriminator.energy(fake, fake_recon).mean()
            pt = _pulling_away_term(fake_embed)

            e_real_sum += e_real.item()
            e_fake_sum += e_fake.item()
            pt_sum += pt.item()
            n_batches += 1

            del z, fake, real_recon, fake_recon, fake_embed, e_real, e_fake, pt

    _clear_runtime_memory()

    if n_batches == 0:
        return {"e_real_eval": 0.0, "e_fake_eval": 0.0, "pt_eval": 0.0}

    return {
        "e_real_eval": e_real_sum / n_batches,
        "e_fake_eval": e_fake_sum / n_batches,
        "pt_eval": pt_sum / n_batches,
    }


def train_ebgan_core(
    generator: Generator,
    discriminator: Discriminator,
    train_loader: DataLoader,
    device: torch.device,
    config: Any,
    start_epoch: int = 1,
    end_epoch: int | None = None,
    opt_g: torch.optim.Optimizer | None = None,
    opt_d: torch.optim.Optimizer | None = None,
    eval_loader: Optional[DataLoader] = None,
 ) -> tuple[list, torch.optim.Optimizer, torch.optim.Optimizer]:
    # Accept dict or object for config
    if isinstance(config, Mapping):
        config = SimpleNamespace(**config)

    if end_epoch is None:
        end_epoch = config.epochs

    if start_epoch < 1 or end_epoch < start_epoch:
        raise ValueError("Invalid epoch range for train_ebgan_core")

    generator.to(device)
    discriminator.to(device)
    generator.train()
    discriminator.train()

    if opt_g is None:
        opt_g = Adam(generator.parameters(), lr=config.lr_g, betas=config.betas)
    if opt_d is None:
        opt_d = Adam(discriminator.parameters(), lr=config.lr_d, betas=config.betas)

    states: list[dict] = []

    for epoch in range(start_epoch, end_epoch + 1):
        g_loss_running = 0.0
        d_loss_running = 0.0
        n_steps = 0
        # training-time metric accumulators
        e_real_sum = 0.0
        e_fake_sum = 0.0
        pt_sum = 0.0
        energy_batches = 0
        pt_batches = 0

        loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch}", unit="batch")
        for batch in loader_iter:
            real_full = _to_device(batch, device)
            batch_size = real_full.size(0)
            real_count, fake_count = _split_real_fake_counts(batch_size, config.real_ratio_for_discriminator)

            for _ in range(config.discriminator_steps):
                real = real_full[:real_count]
                z_d = torch.randn(fake_count, config.latent_dim, 1, 1, device=device)

                with torch.no_grad():
                    fake = generator(z_d)

                real_recon, _ = discriminator(real)
                fake_recon, _ = discriminator(fake)

                e_real = discriminator.energy(real, real_recon).mean()
                e_fake = discriminator.energy(fake, fake_recon).mean()
                e_real_sum += e_real.item()
                e_fake_sum += e_fake.item()
                energy_batches += 1

                d_loss = e_real + F.relu(config.margin - e_fake)

                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

                d_loss_running += d_loss.item()

                del z_d, fake, real_recon, fake_recon, e_real, e_fake, d_loss

            for _ in range(config.generator_steps):
                z_g = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
                fake = generator(z_g)
                fake_recon, fake_embed = discriminator(fake)

                e_fake = discriminator.energy(fake, fake_recon).mean()
                pt = _pulling_away_term(fake_embed)
                g_loss = e_fake + config.pulling_away_weight * pt

                e_fake_sum += e_fake.item()
                pt_sum += pt.item()
                energy_batches += 1
                pt_batches += 1

                opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                opt_g.step()

                g_loss_running += g_loss.item()
                n_steps += 1

                # update tqdm postfix with running averages
                loader_iter.set_postfix(
                    {
                        "g_loss": f"{(g_loss_running / max(n_steps,1)):.4f}",
                        "d_loss": f"{(d_loss_running / max(len(train_loader) * config.discriminator_steps,1)):.4f}",
                    }
                )

                del z_g, fake, fake_recon, fake_embed, e_fake, pt, g_loss

        epoch_state = {
            "epoch": epoch,
            "g_loss": g_loss_running / max(n_steps, 1),
            "d_loss": d_loss_running / max(len(train_loader) * config.discriminator_steps, 1),
            "e_real_train": (e_real_sum / energy_batches) if energy_batches else 0.0,
            "e_fake_train": (e_fake_sum / energy_batches) if energy_batches else 0.0,
            "pt_train": (pt_sum / pt_batches) if pt_batches else 0.0,
        }
        states.append(epoch_state)

        # optionally compute FID at epoch end if requested and eval_loader supplied
        fid_every = int(getattr(config, "compute_fid_every", 0))
        if fid_every and eval_loader is not None and (epoch % fid_every == 0):
            try:
                fid_val = compute_fid(
                    generator,
                    eval_loader,
                    device,
                    latent_dim=int(getattr(config, "latent_dim", 100)),
                    num_batches=int(getattr(config, "fid_num_samples", 10)),
                )
                states[-1]["fid_train"] = float(fid_val)
            except Exception:
                states[-1]["fid_train"] = None

    _clear_runtime_memory()
    return states, opt_g, opt_d



def build_models_from_config(config: Any) -> tuple[Generator, Discriminator]:
    if isinstance(config, Mapping):
        config = SimpleNamespace(**config)

    generator = Generator(
        latent_dim=config.latent_dim,
        activation=config.generator_activation,
        dropout=config.generator_dropout,
        norm=config.generator_norm,
    )

    discriminator = Discriminator(
        embedding_channels=config.discriminator_embedding_channels,
        activation=config.discriminator_activation,
        dropout=config.discriminator_dropout,
        norm=config.discriminator_norm,
    )

    return generator, discriminator


def compute_fid(generator: Generator, dataloader: DataLoader, device: torch.device, latent_dim: int, num_batches: int = 10) -> float:
    """Compute FID using torchmetrics.FrechetInceptionDistance over `num_batches` batches from `dataloader`.

    Images are expected in [-1,1] output from generator; function converts to uint8 [0,255] as required by the metric.
    """
    # Keep the FID / inception model on CPU to avoid loading a large model onto GPU each epoch.
    fid_device = torch.device("cpu")
    fid = FrechetInceptionDistance(feature=2048).to(fid_device)

    generator.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break

            real_imgs = batch[0] if isinstance(batch, (tuple, list)) else batch
            # move real images to CPU for the metric (expecting uint8 [0,255])
            real_imgs_cpu = real_imgs.detach().cpu()

            # Convert to [0,255] uint8 on CPU
            real_imgs_uint8 = ((real_imgs_cpu + 1) * 127.5).clamp(0, 255).to(torch.uint8)

            fid.update(real_imgs_uint8, real=True)

            # generate fake on device, then move to CPU for the metric
            z = torch.randn(real_imgs.size(0), latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)
            fake_imgs_cpu = fake_imgs.detach().cpu()

            fake_imgs_uint8 = ((fake_imgs_cpu + 1) * 127.5).clamp(0, 255).to(torch.uint8)

            fid.update(fake_imgs_uint8, real=False)

    # compute() may use CPU tensors
    result = fid.compute().item()
    # free CPU-side metric state
    del fid
    _clear_runtime_memory()
    return result


def train_ebgan_with_schedule(
    train_loader: DataLoader,
    device: torch.device,
    train_config: Any,
    schedule_config: Any,
    eval_loader: DataLoader | None = None,
) -> dict[str, Any]:
    if isinstance(train_config, Mapping):
        train_config = SimpleNamespace(**train_config)
    if isinstance(schedule_config, Mapping):
        schedule_config = SimpleNamespace(**schedule_config)

    generator, discriminator = build_models_from_config(train_config)
    generator.to(device)
    discriminator.to(device)

    opt_g = Adam(generator.parameters(), lr=train_config.lr_g, betas=train_config.betas)
    opt_d = Adam(discriminator.parameters(), lr=train_config.lr_d, betas=train_config.betas)

    ckpt_dir = Path(schedule_config.checkpoint_dir)
    sample_dir = Path(schedule_config.sample_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)

    fixed_noise = _make_fixed_noise(
        sample_count=schedule_config.sample_count,
        latent_dim=train_config.latent_dim,
        seed=schedule_config.fixed_noise_seed,
        device=device,
    )

    history: list[dict[str, Any]] = []

    for epoch in range(1, train_config.epochs + 1):
        epoch_states, opt_g, opt_d = train_ebgan_core(
            generator=generator,
            discriminator=discriminator,
            train_loader=train_loader,
            device=device,
            config=train_config,
            start_epoch=epoch,
            end_epoch=epoch,
            opt_g=opt_g,
            opt_d=opt_d,
            eval_loader=eval_loader,
        )
        epoch_state = epoch_states[0]
        state_dict = epoch_state

        if eval_loader is not None and epoch % schedule_config.eval_every == 0:
            metrics = evaluate_model(
                generator=generator,
                discriminator=discriminator,
                dataloader=eval_loader,
                device=device,
                latent_dim=train_config.latent_dim,
            )
            state_dict["eval_metrics"] = metrics

            # optional FID at scheduled eval time
            if getattr(schedule_config, "compute_fid", False):
                try:
                    fid_val = compute_fid(
                        generator,
                        eval_loader,
                        device,
                        latent_dim=int(getattr(train_config, "latent_dim", 100)),
                        num_batches=int(getattr(schedule_config, "fid_num_samples", 10)),
                    )
                    state_dict["fid_eval"] = float(fid_val)
                except Exception:
                    state_dict["fid_eval"] = None

            generator.train()
            discriminator.train()

            if schedule_config.cleanup_after_non_training_sampling:
                _clear_runtime_memory()

        if epoch % schedule_config.save_and_sample_every == 0:
            checkpoint_path = ckpt_dir / f"ebgan_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "generator": generator.state_dict(),
                    "discriminator": discriminator.state_dict(),
                    "optimizer_g": opt_g.state_dict(),
                    "optimizer_d": opt_d.state_dict(),
                    "train_config": (vars(train_config) if hasattr(train_config, "__dict__") else dict(train_config)),
                },
                checkpoint_path,
            )

            sample_path = sample_dir / f"sample_epoch_{epoch}.png"
            _save_sample_grid(
                generator=generator,
                fixed_noise=fixed_noise,
                sample_path=sample_path,
                cleanup_after=schedule_config.cleanup_after_non_training_sampling,
            )

            generator.train()
            discriminator.train()

        history.append(state_dict)

    result = {
        "history": history,
        "generator": generator,
        "discriminator": discriminator,
        "optimizer_g": opt_g,
        "optimizer_d": opt_d,
    }

    if schedule_config.unload_models_on_finish:
        clear_model_from_memory(generator, discriminator)
        result["generator"] = None
        result["discriminator"] = None

    return result
