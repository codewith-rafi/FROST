# Project Brief: Fuzzy-Rough Granular Computing for Structure Preservation in One-Step Image Translation

> **About this document.** This is the original FROST project brief. It defines the method, the evaluation plan and the deliverables as they were specified before implementation began. Paths such as `SourceCode/img2img-turbo/` and `src/my_utils/` refer to the planned layout, and the step, checkpoint and dataset numbers are the planned values. The experiments were run under [`../PROTOCOL.md`](../PROTOCOL.md), which lists every deviation from this brief. The as-run code lives in `src/frost/`.

## 1. Objective and Compute Budget

This project evaluates whether a lightweight granular-computing preservation signal improves the structure-versus-appearance trade-off in one-step unpaired image translation without adding an extra inference pass.

Target hardware:

```text
NVIDIA RTX 5090, 32 GB VRAM
```

Primary compute budget:

```text
8 training runs
1,500 optimizer steps per run
12,000 total optimizer steps
fresh output directories
no runtime claim until a 100-step benchmark is completed
```

The 1,500-step budget is exploratory and compute-constrained. It is shorter than the original CycleGAN-Turbo recipe, so underfitting is possible. This limitation must be stated in the final report.

The project compares four conditions:

1. **Baseline CycleGAN-Turbo**: `lambda_granular = 0`.
2. **Binary granular mask**: `lambda_granular = 0.10`.
3. **Fuzzy granular mask**: `lambda_granular = 0.10`.
4. **Fuzzy-rough granular mask**: `lambda_granular = 0.10`.

Datasets:

- Full Horse-to-Zebra.
- Deterministic balanced Day-to-Night subset of BDD100K.

Each trained model is evaluated in both directions:

- Horse to Zebra and Zebra to Horse.
- Day to Night and Night to Day.

Primary metrics:

- **DINO-Struct**: lower is better.
- **FID**: lower is better.

A method is considered better only if both metrics improve. If one improves and the other degrades, report it as a trade-off.

Core constraint:

> Granular masks and the structure-preservation loss are used only during training. Inference remains the original one-step CycleGAN-Turbo procedure.

## 2. Granular Computing Requirement

The project makes granular computing central rather than incidental:

- Image patches are grouped into **granular balls** at two scales.
- **Fuzzy memberships** represent gradual confidence that a patch contains stable structure.
- **Fuzzy-rough lower and upper approximations** distinguish stable structure from ambiguous boundary regions.
- Binary, fuzzy, and fuzzy-rough masks are compared under a controlled protocol.

The project does not claim a new general fuzzy-rough theory. It applies granular-computing principles to a concrete image-translation problem.

## 3. Source-Code Basis

Main repository:

```text
SourceCode/img2img-turbo/
```

Use or create:

```text
src/train_cyclegan_turbo.py
src/my_utils/training_utils.py
src/my_utils/dino_struct.py
src/inference_unpaired.py
src/build_bdd100k_subset.py          # new
src/precompute_granular_masks.py     # new
src/my_utils/granular_masks.py       # new
src/evaluate_unpaired.py             # new
```

Reference repositories:

```text
SourceCode/GBC/
SourceCode/Splice/
```

Use `img2img-turbo` as the training and evaluation backbone. Use GBC as motivation for granular-ball centers, radii, purity, and region grouping. Use Splice only as related work for ViT-based structure preservation.

Do **not** use DINO features in the training loss. DINO-Struct may be used for validation and final evaluation, but it must not enter the generator or discriminator objective.

## 4. Environment, Fresh Restart, and GPU Audit

From `SourceCode/img2img-turbo`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
accelerate config
```

Record:

- GPU model and driver.
- CUDA version.
- Python version.
- PyTorch, Diffusers, Transformers, PEFT, Clean-FID, LPIPS, and xformers versions.
- Source commit hash and local code diff.
- Random seed.
- Exact training command.
- Dataset manifests and image counts.
- Measured steady-state seconds per step.
- Peak VRAM and GPU utilization.

### 4.1 Fresh restart policy

Before launching full runs:

1. Use a new output root:

   ```text
   output/fresh_YYYYMMDD
   ```

2. Do not reuse old checkpoints or old output directories.
3. Confirm no stale trainer is running:

   ```bash
   ps -eo pid,rss,cmd | grep '[t]rain_cyclegan_turbo'
   ```

4. Record GPU state:

   ```bash
   nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv
   ```

5. Record the runtime stack:

   ```bash
   python -c "import torch,diffusers; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_capability(0)); print(diffusers.__version__); print('tf32_matmul', torch.backends.cuda.matmul.allow_tf32); print('tf32_cudnn', torch.backends.cudnn.allow_tf32)"
   ```

6. Record the code snapshot:

   ```bash
   git rev-parse HEAD
   git diff > granular_project_changes.patch
   ```

### 4.2 RTX 5090 attention policy

The `--enable_xformers_memory_efficient_attention` flag may fail on Blackwell GPUs if the installed xformers wheel has no compatible kernel.

Use the same policy for every run:

1. Try xformers in a 100-step smoke test.
2. If it raises a CUDA kernel-image error, omit the flag and use PyTorch native SDPA.
3. Do not switch attention implementations between methods.
4. If xformers succeeds, use it for all eight runs.
5. If xformers fails, use native SDPA for all eight runs.

Native SDPA is acceptable. Do not assume xformers is required.

### 4.3 Required GPU-utilization changes

The current unpaired trainer hardcodes FP32 and does not enable cuDNN TF32. Patch it so `--allow_tf32` enables both:

```python
if args.allow_tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.backends.cudnn.benchmark = True
```

`cudnn.benchmark` is acceptable because the training crop size is fixed at `256 × 256`. It may introduce autotuner nondeterminism, so use the same setting for every run.

Patch the DataLoader:

```python
num_workers = args.dataloader_num_workers

train_dataloader = torch.utils.data.DataLoader(
    dataset_train,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=num_workers > 0,
    prefetch_factor=2 if num_workers > 0 else None,
)
```

Use:

```text
dataloader_num_workers = 4
```

If two concurrent runs are used, this consumes eight CPU workers total. Reduce only if the host has fewer than eight usable CPU cores.

### 4.4 Mixed-precision warning

Do **not** rely on:

```bash
accelerate launch --mixed_precision bf16
```

for the current unpaired trainer. The unpaired parser does not expose `--mixed_precision`, `Accelerator(...)` does not receive it, and the trainer hardcodes:

```python
weight_dtype = torch.float32
```

Therefore, that launch flag does not currently produce a real bf16 CycleGAN training run.

Primary protocol:

```text
FP32 model weights
TF32 enabled for matmul and cuDNN
```

Optional accelerated protocol:

1. Add `--mixed_precision {no,fp16,bf16}` to `parse_args_unpaired_training`.
2. Pass it to `Accelerator`.
3. Remove the hardcoded FP32 dtype.
4. Run a 100-step stability and speed benchmark.
5. If bf16 is stable, use it for all eight runs.
6. Never mix FP32 baseline runs with bf16 granular runs.

Do not add `torch.compile` to the primary experiment unless there is extra time for debugging and all methods use the same setting.

### 4.5 Benchmark and utilization gate

After applying the frequency-condition fixes, run a 100-step benchmark with validation, checkpointing, and visualization disabled:

```bash
accelerate launch --main_process_port 29501 \
  src/train_cyclegan_turbo.py \
  --output_dir output/benchmark_baseline \
  --max_train_steps 100 \
  --validation_steps 0 \
  --checkpointing_steps 0 \
  --viz_freq 0 \
  --validation_num_images 0 \
  --dataloader_num_workers 4 \
  --allow_tf32 \
  --report_to tensorboard
```

Measure:

- Median seconds per step.
- Mean seconds per step.
- Peak VRAM.
- Steady-state GPU utilization.
- Steady-state power draw.

Discard the first 10 steps from the timing calculation because they include warmup.

Required gate:

```text
steady-state GPU utilization: preferably at least 85%
power: generally 350–550 W on an RTX 5090
VRAM: preferably below 30 GB
```

If utilization is below 70%, stop and diagnose CPU dataloading, synchronization overhead, or an incompatible attention kernel before launching the full experiment.

Do not average validation spikes into the training step time.

If the measured steady rate remains approximately 22 seconds per step, the eight 1,500-step runs require roughly 73 hours before evaluation. If two identical GPUs are used, the ideal wall-clock time is approximately halved, but the actual time must still be measured.

### 4.6 Multi-GPU policy

If two GPUs are available, run one training process per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29501 ...
CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29502 ...
```

Do not place two runs on the same GPU. Record which GPU was used for each run. If the GPUs are not identical, do not compare runs trained on different GPUs without stating this limitation.

## 5. Dataset Preparation

### 5.1 Horse-to-Zebra

Use the full local archive:

```text
horse2zebra.zip
```

Standard counts:

| Split | Domain A | Domain B |
|---|---:|---:|
| Train | 1,067 horses | 1,334 zebras |
| Test | 120 horses | 140 zebras |

Use all Horse-to-Zebra images.

The archive uses:

```text
trainA/
trainB/
testA/
testB/
```

The trainer expects:

```text
train_A/
train_B/
test_A/
test_B/
```

Extract and rename from `SourceCode/img2img-turbo`:

```bash
unzip PATH_TO/horse2zebra.zip -d data
cd data/horse2zebra
mv trainA train_A
mv trainB train_B
mv testA test_A
mv testB test_B
```

Create:

```text
data/horse2zebra/fixed_prompt_a.txt
```

Contents:

```text
a horse
```

Create:

```text
data/horse2zebra/fixed_prompt_b.txt
```

Contents:

```text
a zebra
```

Final structure:

```text
SourceCode/img2img-turbo/data/horse2zebra/
├── train_A/
├── train_B/
├── test_A/
├── test_B/
├── fixed_prompt_a.txt
└── fixed_prompt_b.txt
```

### 5.2 BDD100K Day-to-Night subset

Use the local BDD100K package. If the directory is named `bdd100k_pm`, treat it as the BDD100K root.

Expected raw layout:

```text
BDD100K_ROOT/
├── images/
│   └── 100k/
│       ├── train/
│       └── val/
└── labels/
    └── bdd100k_labels_release/
        ├── train.json
        └── val.json
```

Use the official BDD100K training split for training and the official validation split for held-out evaluation. Do not randomly mix train and validation images.

Use the `timeOfDay` attribute.

Keep only:

```text
daytime
night
```

Exclude:

```text
dawn/dusk
undefined
```

#### Deterministic stratified subset

For each domain and split:

1. Filter by `timeOfDay`.
2. Remove duplicate filenames.
3. Group by:

   ```text
   (weather, scene)
   ```

4. Compute a proportional quota for each group.
5. Sort within each group by:

   ```text
   sha256(relative_filename)
   ```

6. Select images according to each group quota.
7. If a group is too small, fill the remaining quota from the global hash-sorted domain list.
8. Save selected filenames and metadata in `manifest.json`.

Use:

- 1,500 daytime training images.
- 1,500 nighttime training images.
- 500 daytime held-out validation images.
- 500 nighttime held-out validation images.

The held-out set improves FID stability while keeping final evaluation manageable.

Create:

```text
src/build_bdd100k_subset.py
```

Example interface:

```bash
python src/build_bdd100k_subset.py \
  --bdd_root PATH_TO/bdd100k_pm \
  --output_dir data/bdd100k_day_night \
  --train_per_domain 1500 \
  --val_per_domain 500 \
  --seed 42
```

Final structure:

```text
SourceCode/img2img-turbo/data/bdd100k_day_night/
├── train_A/
├── train_B/
├── test_A/
├── test_B/
├── fixed_prompt_a.txt
├── fixed_prompt_b.txt
└── manifest.json
```

Where:

```text
train_A = daytime training images
train_B = nighttime training images
test_A  = daytime held-out validation images
test_B  = nighttime held-out validation images
```

Prompt files:

```text
fixed_prompt_a.txt
```

Contents:

```text
a daytime street driving scene
```

```text
fixed_prompt_b.txt
```

Contents:

```text
a nighttime street driving scene
```

### 5.3 Dataset verification

Before training, verify:

- Horse-to-Zebra counts: `1,067 / 1,334 / 120 / 140`.
- BDD100K counts: `1,500 / 1,500 / 500 / 500`.
- Every image opens as RGB.
- No duplicate filename appears in both training and held-out folders.
- No BDD100K image crosses the official train/val boundary.
- Each prompt file has exactly one non-empty line.
- Manifests include image path, domain, split, weather, scene, and time of day.
- Weather and scene distributions are saved.

## 6. Experimental Design

Run eight primary trainings:

| Dataset | Conditions |
|---|---|
| Horse-to-Zebra | Baseline, binary, fuzzy, fuzzy-rough |
| BDD100K Day-to-Night | Baseline, binary, fuzzy, fuzzy-rough |

Evaluate both directions for every model.

Final quantitative table:

```text
2 datasets × 4 methods × 2 directions = 16 entries
```

All methods use the same:

- SD-Turbo checkpoint.
- Backbone and LoRA configuration.
- Dataset split.
- Preprocessing.
- Optimization budget.
- Random seed.
- Validation protocol.
- Attention implementation.
- Precision setting.
- One-step inference procedure.

The only intended difference is the granular preservation signal.

## 7. Streamlined Granular Mask Method

This method is deliberately simple enough to implement and debug in one week while retaining the granular-ball and fuzzy-rough ideas required by the course.

### 7.1 Luminance

Convert RGB to luminance:

```text
Y = 0.299R + 0.587G + 0.114B
```

All mask and loss computations use luminance, not RGB. This prevents the structure signal from forcing the generator to preserve source-domain colors.

### 7.2 Signed-gradient persistence

The training images are normalized to `[-1, 1]` by `UnpairedDataset`. Before computing luminance, rescale:

```text
x_01 = (x + 1) × 0.5
```

Compute Sobel derivatives:

```text
Yx, Yy
```

Compute full-resolution gradient magnitude:

```text
g1 = sqrt(Yx² + Yy²)
```

Compute signed average pooling before magnitude:

```text
Yx_8 = AvgPool(Yx, kernel_size=8, stride=8)
Yy_8 = AvgPool(Yy, kernel_size=8, stride=8)
g8 = sqrt(Yx_8² + Yy_8²)
```

Upsample `g8` to the patch grid or full `288 × 288` resolution.

Do **not** average-pool unsigned gradient magnitude. Signed pooling is required so opposite gradients on zebra stripes can cancel.

For each patch `p`, compute the patch mean of `g1` and `g8`. Define:

```text
persistence_p = clip(4.0 × g8_p / (g1_p + ε), 0, 1)
edge_strength_p = g1_p / (max(g1) + ε)
a_p = clip(persistence_p × sqrt(edge_strength_p), 0, 1)
```

The factor `4.0` compensates for dilution of a thin edge during `8 × 8` pooling. Validate this factor in the mask sanity gate.

This membership should be:

- High for stable object silhouettes.
- Low for zebra stripe interiors.
- Low for grass, foliage, and high-frequency texture.
- Low for ambiguous noise.

### 7.3 Fixed-K granular balls

Compute masks at two scales on a `288 × 288` grid:

| Scale | Patch size | Grid | K |
|---|---:|---:|---:|
| Fine | `8 × 8` | `36 × 36` | 16 |
| Coarse | `16 × 16` | `18 × 18` | 4 |

For each patch, build a 5D descriptor:

```text
d_p = [mean_Y, std_Y, a_p, 0.5 × normalized_x, 0.5 × normalized_y]
```

The spatial coordinates are weighted by `0.5` so k-means does not become a uniform spatial checkerboard.

Standardize each descriptor dimension across the image.

Use deterministic clustering:

```python
MiniBatchKMeans(
    n_clusters=K,
    random_state=42,
    n_init=3
)
```

For each ball `b`, compute:

```text
L_b = P10(a_p inside ball b)
U_b = P90(a_p inside ball b)
B_b = U_b - L_b
mean_b = mean(a_p inside ball b)
```

Where:

- `L_b` is a robust lower approximation.
- `U_b` is a robust upper approximation.
- `B_b` is boundary width.
- `mean_b` is ball-level fuzzy structure.

A ball with high `L_b` and small `B_b` represents stable structure. A ball with large `B_b` represents ambiguous texture or boundary content.

### 7.4 Mask definitions

#### Binary mask

```text
M_binary(p) = 1 if mean_b >= median(mean_b over all balls), else 0
```

The image median prevents an all-zero or all-one mask.

#### Fuzzy mask

```text
M_fuzzy(p) = clip(0.5 × a_p + 0.5 × mean_b, 0, 1)
```

This makes the fuzzy condition granular-ball-based rather than purely patch-based.

#### Fuzzy-rough mask

```text
M_fuzzy_rough(p) =
  clip(a_p × L_b / (U_b + ε) - 0.25 × B_b, 0, 1)
```

This retains patch-level detail while modulating it by the ball’s lower/upper approximation and penalizing boundary ambiguity.

The fuzzy-rough mask should therefore be more selective than the fuzzy mask on zebra stripes, foliage, and other ambiguous texture.

### 7.5 Combine scales

For each mode:

```text
M_final = Resize_286x286(clip(0.6 × M_fine + 0.4 × M_coarse, 0, 1))
```

The official training transform remains `286 × 286`. Padding to `288 × 288` is used only for mask patching.

Save masks at `286 × 286` so they align with the training image transform.

## 8. Offline Mask Precomputation

Do not run k-means inside `UnpairedDataset.__getitem__`.

Precompute masks for:

```text
Horse-to-Zebra: 1,067 + 1,334 = 2,401 training images
BDD100K:        1,500 + 1,500 = 3,000 training images
Total:          5,401 training images
```

Create:

```text
src/precompute_granular_masks.py
```

Example:

```bash
python src/precompute_granular_masks.py \
  --dataset_folder data/horse2zebra \
  --output_dir data/horse2zebra/masks \
  --granular_persistence_scale 4.0 \
  --granular_fine_patch_size 8 \
  --granular_coarse_patch_size 16 \
  --granular_fine_k 16 \
  --granular_coarse_k 4 \
  --granular_lower_percentile 10 \
  --granular_upper_percentile 90 \
  --granular_boundary_penalty 0.25 \
  --granular_seed 42
```

Repeat for BDD100K:

```bash
python src/precompute_granular_masks.py \
  --dataset_folder data/bdd100k_day_night \
  --output_dir data/bdd100k_day_night/masks \
  --granular_persistence_scale 4.0 \
  --granular_fine_patch_size 8 \
  --granular_coarse_patch_size 16 \
  --granular_fine_k 16 \
  --granular_coarse_k 4 \
  --granular_lower_percentile 10 \
  --granular_upper_percentile 90 \
  --granular_boundary_penalty 0.25 \
  --granular_seed 42
```

Save:

```text
data/horse2zebra/masks/{binary,fuzzy,fuzzy_rough}/
data/bdd100k_day_night/masks/{binary,fuzzy,fuzzy_rough}/
```

Use `.pt` float16 tensors or `.npz` arrays. Do not use lossy JPEG storage.

Mask filenames must match image stems.

For qualitative figures, compute masks for selected held-out images only after the qualitative image list has been chosen. Do not use held-out masks to tune the method.

## 9. Aligned Image and Mask Transforms

The official transform is:

```text
resize to 286 × 286
random crop to 256 × 256
random horizontal flip
```

Modify `UnpairedDataset` to use functional transforms so the image and mask receive the same random parameters.

Example:

```python
import torchvision.transforms.functional as TF
import torchvision.transforms as T

crop_top, crop_left, crop_height, crop_width = T.RandomCrop.get_params(
    image, output_size=(256, 256)
)

image = TF.crop(image, crop_top, crop_left, crop_height, crop_width)
mask = TF.crop(mask, crop_top, crop_left, crop_height, crop_width)

if torch.rand(1).item() < 0.5:
    image = TF.hflip(image)
    mask = TF.hflip(mask)
```

Never apply independent random transforms to the image and mask.

Normalize the image to `[-1, 1]`, but keep the mask in `[0, 1]`.

## 10. Fast Coarse-Gradient Structure Loss

For A to B:

```text
input  = img_A
output = fake_B
mask   = M_A
```

For B to A:

```text
input  = img_B
output = fake_A
mask   = M_B
```

Use `fake_A` and `fake_B` already computed by the baseline CycleGAN objective. Do not add another generator forward pass.

### 10.1 Luminance and pooling

1. Rescale input and output from `[-1, 1]` to `[0, 1]`.
2. Convert both to luminance.
3. Average-pool luminance and mask with:

   ```text
   kernel_size=4, stride=4
   ```

### 10.2 Coarse gradient magnitude

On the `4 ×` downsampled luminance maps, compute:

```text
Mag(Y) = sqrt(Sobel_x(Y)² + Sobel_y(Y)² + ε)
```

Normalize each magnitude map by its spatial mean:

```text
Mag_hat(Y) = Mag(Y) / (mean(Mag(Y)) + ε)
```

If the mean magnitude is below `1e-6`, set the normalized map to zero.

This compares relative edge structure while reducing sensitivity to global contrast changes such as Day-to-Night illumination changes.

### 10.3 Masked loss

```text
L_struct(x, G(x), M) =
  mean(M_4 × |Mag_hat(Y_G_4) - Mag_hat(Y_x_4)|)
```

Do **not** divide by `mean(M_4)`. A conservative mask should apply less total preservation force, and a sparse binary mask should not receive an artificially amplified gradient.

Add to the existing generator objective:

```text
L_generator =
  L_GAN_A2B
  + L_GAN_B2A
  + λ_granular × (L_struct_A + L_struct_B)
```

Use:

```text
λ_granular = 0.10
```

The baseline uses:

```text
λ_granular = 0
```

Do not tune `λ_granular` separately for each method in the primary experiment.

## 11. Required Code Changes

### 11.1 Mask module

Create:

```text
src/my_utils/granular_masks.py
```

Provide:

```python
compute_structural_statistics(image)
build_granular_balls(statistics)
derive_masks(statistics, mode)
```

### 11.2 Training arguments

Add to `parse_args_unpaired_training`:

```text
--granular_mode {none,binary,fuzzy,fuzzy_rough}
--lambda_granular
--granular_mask_dir
--granular_persistence_scale
--granular_fine_patch_size
--granular_coarse_patch_size
--granular_fine_k
--granular_coarse_k
--granular_lower_percentile
--granular_upper_percentile
--granular_boundary_penalty
--granular_seed
```

Defaults:

```text
granular_mode = none
lambda_granular = 0.10
granular_persistence_scale = 4.0
granular_fine_patch_size = 8
granular_coarse_patch_size = 16
granular_fine_k = 16
granular_coarse_k = 4
granular_lower_percentile = 10
granular_upper_percentile = 90
granular_boundary_penalty = 0.25
granular_seed = 42
```

### 11.3 Dataset changes

Modify `UnpairedDataset` to:

1. Load the image.
2. Load the corresponding precomputed mask.
3. Apply the same random crop and horizontal flip to both.
4. Return:

   ```text
   granular_mask_src
   granular_mask_tgt
   ```

For baseline mode, skip mask loading to avoid unnecessary I/O. If the collate function requires a mask, return unit masks and keep `lambda_granular = 0`.

### 11.4 Training changes

Modify `src/train_cyclegan_turbo.py` to:

1. Read source and target masks.
2. Compute the coarse-gradient structure loss.
3. Add it to the generator objective.
4. Log:

   ```text
   train/loss_granular_a2b
   train/loss_granular_b2a
   train/mask_mean_src
   train/mask_mean_tgt
   ```

Do not use DINO in this loss.

### 11.5 Checkpoint, validation, and visualization fixes

The public script uses conditions that can miss the final step. Change checkpointing to:

```python
if args.checkpointing_steps > 0 and (
    global_step % args.checkpointing_steps == 0
    or global_step == args.max_train_steps
):
    save_checkpoint()
```

Change validation to:

```python
if args.validation_steps > 0 and (
    global_step % args.validation_steps == 0
    or global_step == args.max_train_steps
):
    run_validation()
```

Change visualization to:

```python
if args.viz_freq > 0 and global_step % args.viz_freq == 0:
    visualize()
```

The `> 0` guards are required so a benchmark can disable validation, checkpointing, and visualization by passing `0`.

Change:

```python
if idx > args.validation_num_images
```

to:

```python
if idx >= args.validation_num_images
```

With `checkpointing_steps=750` and `max_train_steps=1500`, the trainer should save:

```text
model_750.pkl
model_1500.pkl
```

Only `model_1500.pkl` is used for final evaluation.

### 11.6 Final evaluation script

Create:

```text
src/evaluate_unpaired.py
```

It should:

1. Load `model_1500.pkl`.
2. Translate all held-out images once per direction.
3. Compute Clean-FID.
4. Compute DINO-Struct.
5. Save per-image and aggregate metrics.
6. Record dataset, method, direction, checkpoint, seed, precision setting, and attention implementation.
7. Reuse cached FID reference statistics for each dataset and direction when possible.

Use the fixed final checkpoint. Do not select the best checkpoint using held-out results.

## 12. Mask Sanity Gate

Before full training, inspect masks on training images only:

- 20 zebras.
- 20 horses.
- 20 BDD daytime images.
- 20 BDD nighttime images.

Required checks:

1. Zebra stripe interiors have lower mask values than object silhouettes.
2. Horse body interiors are not strongly protected when the target is zebra.
3. Road and building boundaries receive higher weight than foliage or sky texture.
4. Binary masks are not nearly all zeros or all ones.
5. Fuzzy-rough masks are lower than fuzzy masks on ambiguous texture.
6. Stable object boundaries remain visible in the fuzzy-rough mask.

Report:

```text
mean mask value
10th percentile
50th percentile
90th percentile
fraction M > 0.75
fraction 0.25 < M <= 0.75
fraction M <= 0.25
```

For fuzzy-rough masks, also report:

```text
lower-approximation fraction
boundary fraction
outside-upper-approximation fraction
mean boundary width B_b
```

If stable silhouettes are too dim, adjust only:

```text
--granular_persistence_scale
```

and rerun the sanity gate before training. Do not tune masks after observing held-out results.

## 13. Training Protocol

Use the same settings for all eight runs:

```text
pretrained model = stabilityai/sd-turbo
train batch size = 1
gradient accumulation = 1
learning rate = 1e-5
max train steps = 1,500
random seed = 42
train image prep = resize_286_randomcrop_256x256_hflip
validation image prep = no_resize
UNet LoRA rank = 128
VAE LoRA rank = 4
validation steps = 1,500
checkpointing steps = 750
validation images = 20
viz frequency = 1,500
dataloader workers = 4
TF32 = enabled
gradient checkpointing = disabled
lambda_gan = 0.5
lambda_idt = 1.0
lambda_cycle = 1.0
lambda_cycle_lpips = 10.0
lambda_idt_lpips = 1.0
```

Training validation uses only 20 images to reduce overhead. Final evaluation uses all held-out images.

Do not use early stopping. Train every model for the full 1,500 steps and evaluate `model_1500.pkl`.

### 13.1 Baseline command template

Run from `SourceCode/img2img-turbo`:

```bash
accelerate launch --main_process_port 29501 \
  src/train_cyclegan_turbo.py \
  --pretrained_model_name_or_path stabilityai/sd-turbo \
  --output_dir output/fresh_YYYYMMDD/DATASET/baseline \
  --dataset_folder data/DATASET_FOLDER \
  --train_img_prep resize_286_randomcrop_256x256_hflip \
  --val_img_prep no_resize \
  --learning_rate 1e-5 \
  --max_train_steps 1500 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --seed 42 \
  --report_to tensorboard \
  --tracker_project_name granular_cycle_turbo \
  --validation_steps 1500 \
  --checkpointing_steps 750 \
  --validation_num_images 20 \
  --viz_freq 1500 \
  --dataloader_num_workers 4 \
  --allow_tf32 \
  --lambda_gan 0.5 \
  --lambda_idt 1 \
  --lambda_cycle 1 \
  --lambda_cycle_lpips 10 \
  --lambda_idt_lpips 1 \
  --granular_mode none \
  --lambda_granular 0
```

Add:

```text
--enable_xformers_memory_efficient_attention
```

only if the RTX 5090 smoke test succeeds.

Do not add:

```text
--gradient_checkpointing
```

unless every run uses it after an OOM. Gradient checkpointing saves VRAM but usually slows training.

### 13.2 Proposed-run substitutions

For binary, fuzzy, and fuzzy-rough runs, add:

```text
--granular_mode MODE
--granular_mask_dir data/DATASET_FOLDER/masks
--lambda_granular 0.10
```

Modes:

```text
binary
fuzzy
fuzzy_rough
```

Dataset substitutions:

| Dataset | Dataset folder | Mask directory |
|---|---|---|
| Horse-to-Zebra | `data/horse2zebra` | `data/horse2zebra/masks` |
| BDD100K | `data/bdd100k_day_night` | `data/bdd100k_day_night/masks` |

Output directories:

```text
output/fresh_YYYYMMDD/horse2zebra/baseline
output/fresh_YYYYMMDD/horse2zebra/binary
output/fresh_YYYYMMDD/horse2zebra/fuzzy
output/fresh_YYYYMMDD/horse2zebra/fuzzy_rough

output/fresh_YYYYMMDD/bdd100k_day_night/baseline
output/fresh_YYYYMMDD/bdd100k_day_night/binary
output/fresh_YYYYMMDD/bdd100k_day_night/fuzzy
output/fresh_YYYYMMDD/bdd100k_day_night/fuzzy_rough
```

If two GPUs are used, assign one run per GPU and use different `--main_process_port` values.

## 14. Inference

Use:

```text
src/inference_unpaired.py
```

Example Horse to Zebra:

```bash
python src/inference_unpaired.py \
  --model_path output/fresh_YYYYMMDD/horse2zebra/fuzzy_rough/checkpoints/model_1500.pkl \
  --input_image data/horse2zebra/test_A/n02381460_20.jpg \
  --prompt "a zebra" \
  --direction a2b \
  --output_dir outputs/fresh_YYYYMMDD/horse2zebra/fuzzy_rough/a2b \
  --image_prep no_resize
```

Example Zebra to Horse:

```bash
python src/inference_unpaired.py \
  --model_path output/fresh_YYYYMMDD/horse2zebra/fuzzy_rough/checkpoints/model_1500.pkl \
  --input_image data/horse2zebra/test_B/n02381460_20.jpg \
  --prompt "a horse" \
  --direction b2a \
  --output_dir outputs/fresh_YYYYMMDD/horse2zebra/fuzzy_rough/b2a \
  --image_prep no_resize
```

Prompt rules:

- A to B uses `fixed_prompt_b.txt`.
- B to A uses `fixed_prompt_a.txt`.

Every model performs exactly one diffusion step.

## 15. Final Evaluation

### 15.1 FID

Use Clean-FID.

For A to B:

```text
generated images = translations from test_A
reference images = real test_B
```

For B to A:

```text
generated images = translations from test_B
reference images = real test_A
```

Use all held-out images.

Cache reference statistics per dataset and direction when possible. Do not recompute identical reference statistics for every method if it can be avoided.

### 15.2 DINO-Struct

Use:

```text
src/my_utils/dino_struct.py
```

The metric uses:

- DINO ViT-B/8.
- Layer 11.
- Key self-similarity maps.
- MSE between input and output self-similarity maps.

Report:

- Mean.
- Standard deviation.
- Per-image values.

DINO is used only for evaluation, not for the training loss.

### 15.3 Qualitative evaluation

For each dataset and direction, choose six held-out images before viewing outputs.

Show:

1. Input.
2. Baseline output.
3. Binary output.
4. Fuzzy output.
5. Fuzzy-rough output.
6. Real target reference.
7. Fuzzy-rough mask.

Use the same image order for every method.

### 15.4 Mask diagnostics

Report:

```text
mean mask value
10th percentile
50th percentile
90th percentile
fraction M > 0.75
fraction 0.25 < M <= 0.75
fraction M <= 0.25
```

For fuzzy-rough masks, also report:

```text
lower-approximation fraction
boundary fraction
outside-upper-approximation fraction
mean boundary width B_b
```

Use training masks for these diagnostics. Do not tune the method using held-out masks.

## 16. Statistical Analysis

Primary protocol:

```text
seed = 42
one run per method
```

For DINO-Struct:

- Compute paired per-image differences against the baseline.
- Use paired bootstrap with at least 1,000 resamples.
- Report 95% confidence intervals.

For FID:

- Report point estimates.
- If compute permits, bootstrap generated and reference feature sets.
- Do not claim per-image significance because FID is a global distribution metric.

Do not claim statistical significance from one seed.

## 17. Result Interpretation

Create one table per dataset:

| Method | A→B FID | A→B DINO | B→A FID | B→A DINO |
|---|---:|---:|---:|---:|
| Baseline |  |  |  |  |
| Binary |  |  |  |  |
| Fuzzy |  |  |  |  |
| Fuzzy-rough |  |  |  |  |

Interpretation:

1. **Better overall**: FID and DINO both decrease.
2. **Structure trade-off**: DINO decreases, FID increases.
3. **Distribution trade-off**: FID decreases, DINO increases.
4. **Worse overall**: both increase.

Discuss whether fuzzy-rough separation is more selective than binary or fuzzy weighting and whether effects transfer from Horse-to-Zebra to BDD100K.

## 18. Seven-Day Execution Plan

### Day 1

- Create a fresh environment.
- Prepare Horse-to-Zebra.
- Build the BDD100K subset and manifest.
- Verify all image counts.
- Apply GPU, DataLoader, checkpoint, and validation fixes.
- Run the GPU audit and 100-step baseline benchmark.

### Day 2

- Implement `granular_masks.py`.
- Implement `precompute_granular_masks.py`.
- Precompute masks for both datasets.
- Run the mask sanity gate.

### Day 3

- Modify `UnpairedDataset` for aligned transforms.
- Add the granular training arguments and structure loss.
- Run short smoke tests for all four modes.
- Run a 100-step benchmark for baseline and fuzzy-rough.
- Confirm the GPU-utilization gate.

### Day 4

- Launch the Horse-to-Zebra runs.
- If two GPUs are available, launch BDD100K runs in parallel on the second GPU.
- Record step time, VRAM, and utilization for every run.

### Day 5

- Finish remaining Horse-to-Zebra and BDD100K runs.
- Begin final inference on completed `model_1500.pkl` checkpoints.

### Day 6

- Finish final FID and DINO-Struct evaluation.
- Build quantitative tables.
- Build qualitative comparison grids.
- Compute mask diagnostics.

### Day 7

- Write the 6–8 page report.
- Verify references, figures, tables, commands, and limitations.
- Prepare the oral presentation.

If time becomes critical, prioritize completing all four Horse-to-Zebra methods before BDD100K. Do not selectively increase steps or tune one method more than another.

## 19. Final Report Structure

Use the course-required 6–8 page structure:

1. **Title and authors**
2. **Abstract**
3. **Introduction**
4. **Related Work**
5. **Data**
6. **Methods**
7. **Experiments**
8. **Conclusion**

Include:

- Dataset counts and manifests.
- Granular-ball construction.
- Binary, fuzzy, and fuzzy-rough mask definitions.
- Signed-gradient persistence.
- Structure-loss definition.
- Training protocol.
- GPU and precision settings.
- Measured runtime.
- FID and DINO-Struct tables.
- Mask diagnostics.
- Qualitative comparisons.
- Trade-off analysis.
- Limitations.

Limitations to state:

- 1,500 steps is a compute-constrained budget and may underfit.
- One seed is used.
- FID remains noisy on moderate-size held-out sets.
- Masks are heuristic.
- The method is evaluated on two datasets only.
- Improvement is not guaranteed.
- The RTX 5090 speedup depends on the measured benchmark, not on an assumed runtime.

## 20. Deliverables

1. Dataset manifests.
2. Exact training commands.
3. Precomputed mask directories.
4. Eight final `model_1500.pkl` checkpoints.
5. Generated images for both directions.
6. FID table.
7. DINO-Struct table.
8. Per-image DINO values.
9. Mask diagnostics.
10. Qualitative comparison figures.
11. GPU benchmark and runtime table.
12. Final 6–8 page report.
13. Oral presentation.
14. Source-code and paper citations.

## 21. Pre-Launch Checklist

Before launching full runs, confirm:

- No stale training process is running.
- A fresh output root is used.
- Horse-to-Zebra counts are correct.
- BDD100K counts are correct.
- BDD100K uses the official train/val boundary.
- Prompt files are correct.
- Masks are precomputed.
- Mask sanity gate passed.
- Image and mask transforms are aligned.
- TF32 matmul and cuDNN are enabled.
- DataLoader workers, prefetching, and persistent workers are enabled.
- Checkpoint, validation, visualization, and image-count off-by-one bugs are fixed.
- The 100-step benchmark passed the GPU-utilization gate.
- The same attention implementation is used for all runs.
- The same precision setting is used for all runs.
- The exact command for every run is saved.

## 22. References

1. Parmar, G., Park, T., Narasimhan, S., & Zhu, J.-Y. (2024). *One-Step Image Translation with Text-to-Image Models*. arXiv:2403.12036.

2. Lian, X., Xia, S., Sang, B., Wang, G., & Gao, X. (2026). *GBFRS: Robust Fuzzy Rough Sets via Granular Ball Computing*. IEEE Transactions on Neural Networks and Learning Systems. https://doi.org/10.1109/TNNLS.2026.3681370

3. Wen, X., Fan, Y., Sun, F., Zhang, X., Sun, K., Xiao, H., & Liu, P. (2026). *A Novel Model of Fuzzy Rough Set with Applications in Data Classification and Image Segmentation*. Applied Soft Computing, 191, 114677. https://doi.org/10.1016/j.asoc.2026.114677

4. Xu, Z., Zhai, H., Zeng, Z., Lin, B., & Deng, M. (2026). *GBGCN: Adaptive granular-ball graph representation and clarity-aware GCN for multi-focus image fusion*. Knowledge-Based Systems, 336, 115271. https://doi.org/10.1016/j.knosys.2026.115271

5. Lu, L., Guo, M., Chen, X., & Shi, Y. (2026). *Semantic Granularity Navigation in Image Editing*. Proceedings of the 43rd International Conference on Machine Learning, PMLR 306. https://arxiv.org/abs/2605.21190

6. Song, T., Wen, R., & Zhang, L. (2024). *RoughSet-DDPM: An Image Super-Resolution Method Based on Rough Set Denoising*. Technical Gazette, 31(1), 162–170. https://doi.org/10.17559/TV-20230717000808

7. Tumanyan, N., Bar-Tal, O., Bagon, S., & Dekel, T. (2022). *Splicing ViT Features for Semantic Appearance Transfer*. CVPR.