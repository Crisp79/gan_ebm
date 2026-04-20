# EBGAN Implementation Plan (Phased)

## Phase 1: Architecture (`dgm.py` and `dem.py`)

### 1.1 `dgm.py` (Generator)
- Build `Generator` that maps latent noise `z` to `64 x 64` images.
- Use `ConvTranspose2d` upsampling blocks.
- Default architecture path uses:
    - Batch Normalization
    - ReLU activations
    - final Tanh output in `[-1, 1]`
- Keep existing configurability intact through config-controlled options:
    - generator activation
    - generator dropout
    - generator normalization mode (`batchnorm` or `none`)

### 1.2 `dem.py` (Autoencoder / Discriminator)
- Build `EnergyModel`-style discriminator with explicit `Encoder` and `Decoder` modules.
- Encoder requirements:
    - downsample image to bottleneck `S` (configurable embedding channels)
    - spectral normalization on convolutional layers
    - Layer Normalization (not BatchNorm)
    - LeakyReLU nonlinearity
- Decoder requirements:
    - upsample bottleneck back to `64 x 64`
    - final Tanh output
- Forward output requirements:
    - reconstructed image
    - bottleneck `S`

## Phase 2: Math and Data (`utils.py`)

### 2.1 Data Loading
- Add Dataset and DataLoader orchestration for the glasses vs no-glasses data.
- Ensure DataLoader defaults are Windows-safe:
    - `num_workers=0` by default
- Keep transforms and batch collation compatible with current training pipeline.

### 2.2 Energy Function
- Implement helper for per-sample reconstruction energy using MSE:
    - `E(x) = MSE(x, x_hat)`

### 2.3 Pulling-away Term
- Implement `compute_pt(S)`:
    - flatten bottleneck vectors per sample
    - apply L2 normalization
    - compute cosine similarity matrix
    - return PT penalty from off-diagonal similarities

## Phase 3: Execution Pipeline (`train_ebgan.py`)

### 3.1 Initialization
- Instantiate `G` from `dgm.py` and `D` from `dem.py`.
- Use separate Adam optimizers for G and D.
- Use EBGAN hyperparameters:
    - margin `m` (initially in 10-20 range)
    - PT weight `lambda` (initially around 0.1)

### 3.2 Phase 1 (Train DEM)
- Compute `E_real` from real image reconstructions.
- Compute `E_fake` from detached fake image reconstructions.
- Optimize discriminator with hinge objective:
    - `L_D = E_real + max(0, m - E_fake)`

### 3.3 Phase 2 (Train DGM)
- Generate fresh fake images.
- Pass through DEM to obtain reconstructions and bottleneck `S`.
- Compute `E_fake` and `PT(S)`.
- Optimize generator with:
    - `L_G = E_fake + lambda * PT`

### 3.4 Logging and Checkpointing
- Track and log:
    - `E_real`
    - `E_fake`
    - `PT`
- Checkpoint state dicts every few epochs.
- Keep scheduled behavior:
    - evaluate every 5 epochs
    - save and sample every 15 epochs

## Config Compatibility (Must Stay Same)

The config contract already used in `train_ebgan.py` must remain unchanged (no removals or renames).

### TrainConfig keys to preserve
- `epochs`
- `latent_dim`
- `lr_g`
- `lr_d`
- `betas`
- `generator_steps`
- `discriminator_steps`
- `real_ratio_for_discriminator`
- `generator_activation`
- `discriminator_activation`
- `generator_dropout`
- `discriminator_dropout`
- `generator_norm`
- `discriminator_norm`
- `margin`
- `pulling_away_weight`
- `discriminator_embedding_channels`

### ScheduleConfig keys to preserve
- `eval_every`
- `save_and_sample_every`
- `checkpoint_dir`
- `sample_dir`
- `sample_count`
- `fixed_noise_seed`
- `cleanup_after_non_training_sampling`
- `unload_models_on_finish`

## Memory Management Requirement

After sampling/evaluation sections where training is not immediately continuing:
- release temporary tensors/objects
- run `gc.collect()`
- if CUDA is available, run:
    - `torch.cuda.empty_cache()`
    - `torch.cuda.ipc_collect()`

When models should be unloaded after final sampling/checkpointing, call explicit model cleanup.

## Acceptance Criteria

- Phase 1, 2, and 3 tasks are reflected in code structure and interfaces.
- `dgm.py` and `dem.py` expose generator and autoencoder energy model roles clearly.
- `utils.py` contains dataset/data loader defaults plus energy/PT helpers.
- `train_ebgan.py` uses two-phase EBGAN optimization with requested losses.
- Existing config keys remain backward compatible.