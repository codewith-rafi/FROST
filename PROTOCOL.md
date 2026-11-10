# FROST Experimental Protocol

This document records the protocol under which all eight FROST runs were trained and evaluated. The scientific specification is the original project brief, [`docs/instruction.md`](docs/instruction.md). Where the two documents differ, **this protocol is authoritative**. Every deviation from the brief, such as the step count, checkpoint interval and BDD100K subset size noted in the tables below, was fixed before the main runs and applied identically to all methods.

## 1. Execution environment

All runs were executed on a shared lab host. The following settings were held fixed:

| Item | Value |
|---|---|
| Python environment | conda env `rafi` (Python 3.11.16) |
| Training GPU | GPU 1 only |
| Final-evaluation GPU | GPU 0 |
| Hardware | 2× NVIDIA RTX 5090 32 GB, driver 595.84, CUDA 13.2, torch 2.10.0+cu128 |
| Process management | one named `tmux` session per job; all output is `tee`'d into the run's log |
| Experiment tracking | Weights & Biases in offline mode (`WANDB_MODE=offline`) |
| Model downloads | `HF_ENDPOINT=https://hf-mirror.com` (overridable) |

The launchers in `scripts/` encode these settings. To reproduce on different hardware, change the conda activation line and `CUDA_VISIBLE_DEVICES` in the launchers, or run the Python modules (`frost.train_cyclegan_turbo`, `frost.evaluate_unpaired`) directly with the arguments the launchers pass.

**Single-GPU training.** Each run trains on exactly one GPU and no two runs ever share a GPU. All eight training runs used the same GPU. Evaluation on a second, identical GPU does not affect the comparison between methods.

## 2. Training budget

| Item | Value |
|---|---|
| Optimizer steps | **2,000** per run (brief: 1,500) |
| Checkpoints | every **500** steps; only **`model_2000.pkl`** is evaluated |
| In-loop validation | disabled (`validation_steps=2000`, `validation_num_images=0`); final evaluation runs offline on **all** held-out images |
| Visualisation | disabled (`viz_freq=0`) |
| DataLoader workers | Horse ↔ Zebra: **8**. BDD100K: **0** (worker processes crashed repeatedly on the BDD100K runs) |
| Runs | **8** = Horse ↔ Zebra × 4 methods + BDD100K × 4 methods |
| Seed (training and masks) | **42** |
| Attention | native PyTorch SDPA; xformers **off** |
| TF32 | **on** for matmul and cuDNN; `cudnn.benchmark` on |
| Precision | **FP32** weights |
| `lambda_granular` | baseline **0**; binary, fuzzy and fuzzy-rough **0.10** |
| Output root | `output/fresh_YYYYMMDD/<dataset>/<method>/`; existing non-empty checkpoint directories are never reused |

Measured throughput was about 20–22 s per step, or roughly 11–12.5 h per 2,000-step run. See [`report/tables/runtime_protocol.md`](report/tables/runtime_protocol.md).

**Why FP32.** The trainer hard-codes `weight_dtype = torch.float32` and does not cast the models. Passing `accelerate launch --mixed_precision bf16` would therefore not produce a genuine bf16 run, so the flag is not used.

**Why SDPA instead of xformers.** xformers 0.0.35 provides no kernel for the RTX 5090 (sm_120). Diffusers and PyTorch fall back to `AttnProcessor2_0` (native SDPA), which needs no extra kernel. The same attention implementation is used for every run.

## 3. Datasets

| Dataset | Train A / B | Test A / B | Notes |
|---|---|---|---|
| Horse ↔ Zebra | 1,067 / 1,334 | 120 / 140 | full standard split, frozen |
| BDD100K Day ↔ Night | **1,067 / 1,334** | **120 / 140** | counts matched to Horse ↔ Zebra; subset seed **37** (brief: 1,500/1,500/500/500, seed 42) |

To rebuild the BDD subset:

```bash
python scripts/build_bdd_day_night_subset.py \
  --train-day 1067 --train-night 1334 --val-day 120 --val-night 140 --seed 37
```

**Dataset location.** Datasets live in `<project root>/data/<dataset>/`. Configs pass `--dataset_folder data/<dataset>` relative to the project root, and every launcher uses the project root as its working directory. On the training host, the `train_*` / `test_*` entries are absolute symlinks into the raw trees (`100k/` and `horse2zebra/`). The `data/` folder can therefore be moved, but the raw trees cannot. The full layout is described in [`data/readme.md`](data/readme.md).

## 4. Code organisation

- **First-party code:** [`src/frost/`](src/frost/) contains the masks, the structure loss, the dataset, the CLI, the trainer, mask precomputation and evaluation.
- **Vendored backbone (import-only):** [`vendor/img2img-turbo/`](vendor/img2img-turbo/) provides `cyclegan_turbo`, `model` and `dino_struct`. No new project logic is added under `vendor/`; see [`docs/vendor_manifest.md`](docs/vendor_manifest.md).
- **Entry point:** `accelerate launch -m frost.train_cyclegan_turbo`, run with `PYTHONPATH=<root>/src:<root>/vendor/img2img-turbo/src`.
- **Configs:** [`configs/course/`](configs/course/). **Scripts:** [`scripts/`](scripts/). **Manifests:** [`manifests/`](manifests/).

## 5. Granular masks

Fixed-K masks (`granular_seed=42`) are precomputed for the training splits into `data/<dataset>/masks/{train_A,train_B}/<stem>_mask_<mode>.npz`:

```bash
bash scripts/precompute_masks.sh horse2zebra
bash scripts/precompute_masks.sh bdd100k_day_night
python scripts/mask_diagnostics.py --all --masks_root data --out_dir report/tables/mask_diagnostics
```

Mask diagnostics are computed on training masks only. Held-out masks exist only for the pre-registered qualitative images (`report/masks_heldout/`) and were never used for tuning.

## 6. Launching runs

```bash
# one run
bash scripts/launch_train.sh configs/course/horse2zebra_baseline.yaml

# all eight runs (Horse ↔ Zebra, then BDD100K); non-baseline methods require precomputed masks
bash scripts/run_course_primary.sh
```

The launcher copies the config into the run directory and writes the exact command to `train.log` as a `TRAIN_CMD ...` line. Copying that line replays the run.

## 7. Final evaluation

```bash
bash scripts/eval_checkpoint.sh <course_yaml> <run>/checkpoints/model_2000.pkl
```

This computes Clean-FID and DINO-Struct (mean, standard deviation and per-image values) in both directions on all held-out images, and writes `metrics.json` into the run directory. `scripts/run_course_eval.sh` evaluates all eight runs in sequence and skips any run that already has a valid `metrics.json`.
