# GPU, protocol and measured runtime (§20.11)

Built from `TRAIN_END` / `EVAL_START` / `EVAL_END` lines and tqdm rates in the run logs. Nothing here was re-executed.

## Training (measured wall clock)

| Dataset | Method | Steps (executed) | Wall | s/step | final_step | note |
|---|---|---:|---:|---:|---:|---|
| horse | baseline | 2000 | 11:38:20 | 20.95 | 2000 |  |
| horse | binary | 2000 | 11:38:39 | 20.96 | 2000 |  |
| horse | fuzzy | 2000 | 11:29:55 | 20.70 | 2000 |  |
| horse | fuzzy_rough | 2000 | 11:14:13 | 20.23 | 2000 |  |
| bdd | baseline | 2000 | 12:01:22 | 21.64 | 2000 |  |
| bdd | binary | 2000 | 03:48:57 | 6.87 | 2000 | 3.1x faster than every other run at identical steps; cause not instrumented, not a speedup claim |
| bdd | fuzzy | 500 | 03:06:06 | 22.33 | 2000 | elapsed covers only the resumed leg (model_1500 + this leg) |
| bdd | fuzzy_rough | 2000 | 12:24:23 | 22.33 | 2000 |  |

## Benchmark (100-step gate)

- `util_gate_100`: wall 01:13:37 over **199** steps = 22.20 s/step.
- Caveat: the run was configured for 100 steps but reached `final_step=199` because of the pre-fix epoch-break bug, so it is a 199-step measurement, not 100. Its s/step is consistent with the 2000-step runs.

## Evaluation (generation vs FID phases)

| Dataset | Method | Wall | a2b gen | b2a gen | FID phase (wall - gen) | note |
|---|---|---:|---:|---:|---:|---|
| horse | baseline | 132 min | 13.53 s/img | 16.30 s/img | 67 min |  |
| horse | binary | 95 min | 7.87 s/img | 10.76 s/img | 54 min |  |
| horse | fuzzy | 96 min | 11.31 s/img | 11.19 s/img | 47 min |  |
| horse | fuzzy_rough | 96 min | 11.95 s/img | 9.85 s/img | 49 min |  |
| bdd | baseline | 147 min | 17.79 s/img | 18.55 s/img | 68 min |  |
| bdd | binary | 599 min | 22.66 s/img | 22.98 s/img | n/a | wall (599 min) contaminated by the Clean-FID Inception download stall (~3 h) plus kill and relaunch, so no per-phase split is possible; generation rates above come from the completed tqdm bars and are still valid |
| bdd | fuzzy | 194 min | 22.17 s/img | 24.12 s/img | 94 min |  |
| bdd | fuzzy_rough | 197 min | 23.76 s/img | 26.36 s/img | 88 min |  |

Generation rates are descriptive: they vary run to run with GPU contention (a concurrent job shared the host) and were not a controlled variable.

## Protocol held constant across all eight runs

| Item | Value |
|---|---|
| Backbone | `stabilityai/sd-turbo`, one diffusion step |
| GPU (training) | GPU 1, `CUDA_VISIBLE_DEVICES=1` (from `scripts/launch_train.sh`) |
| GPU (final eval) | GPU 0 (GPU 1 was occupied by another job during the evaluation window) |
| Precision | FP32 weights + TF32 matmul and cuDNN |
| Attention | native PyTorch SDPA; xformers never enabled |
| Batch / accum | 1 / 1 |
| LoRA rank | UNet 128, VAE 4 |
| Learning rate | 1e-5, `constant` schedule |
| Steps | 2000 (`max_train_epochs=1`) |
| Checkpointing | every 500 |
| Validation in-loop | disabled (`validation_num_images=0`) |
| Seed | 42 |
| DataLoader workers | 8 (horse), 0 (BDD) |
| Hardware | 2x RTX 5090 32 GB, driver 595.84, CUDA 13.2, torch 2.10.0+cu128, python 3.11.16 |
