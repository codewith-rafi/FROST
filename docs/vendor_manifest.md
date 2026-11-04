# Vendored Code: `vendor/img2img-turbo/`

FROST includes a **pinned, minimal slice** of
[`img2img-turbo`](https://github.com/GaParmar/img2img-turbo), the repository that provides the CycleGAN-Turbo backbone.
All project logic lives in [`../src/frost/`](../src/frost/). The vendor tree is **import-only**.

## Provenance

| Field | Value |
|---|---|
| Upstream remote | `https://github.com/GaParmar/img2img-turbo` |
| Upstream branch | `main` |
| Upstream HEAD | `86f54146590f` |
| Snapshot taken | 2026-11-02 |
| License | MIT; see [`../vendor/img2img-turbo/LICENSE`](../vendor/img2img-turbo/LICENSE) |

The original clone carried local modifications. They are preserved as a unified diff in
[`../report/tables/vendor_patches/img2img-turbo-content.patch`](../report/tables/vendor_patches/img2img-turbo-content.patch).

## Retained files

| File | Imported by |
|---|---|
| `src/model.py` | `cyclegan_turbo.py`, `frost.train_cyclegan_turbo` — `make_1step_sched`, VAE forward hooks, `download_url` |
| `src/cyclegan_turbo.py` | `frost.train_cyclegan_turbo`, `frost.evaluate_unpaired` |
| `src/my_utils/dino_struct.py` | train + eval (DINO-Struct metric) |
| `src/inference_unpaired.py` | single-image inference (instruction §14) |
| `src/my_utils/training_utils.py` | imported by `inference_unpaired.py` for `build_transform`; also carries the granular CLI arguments |
| `requirements.txt` | upstream dependency reference |
| `LICENSE` | attribution |

`src/my_utils/` intentionally has **no** `__init__.py`. It works as a namespace package, and the
launchers put `vendor/img2img-turbo/src` on `PYTHONPATH`.

## Local changes to vendor code

1. **Granular CLI arguments** in `src/my_utils/training_utils.py` — captured in the patch file above.
2. **xformers guard** in `src/inference_unpaired.py`: the upstream call
   `model.unet.enable_xformers_memory_efficient_attention()` is now wrapped in `try/except`.
   RTX 5090 (sm_120) has no compatible xformers kernel, so the unguarded call made the script
   unusable. If the call fails, the script now falls back to native SDPA.

## Data paths

Datasets live at the **project root** under `data/`, not inside the vendor tree:

```text
<project root>/data/{horse2zebra,bdd100k_day_night}/{train_A,train_B,test_A,test_B,masks}
```

Launchers run with the project root as the working directory and pass
`--dataset_folder data/<dataset>`. See [`../data/readme.md`](../data/readme.md) for the full layout.
