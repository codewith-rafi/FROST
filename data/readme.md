# `data/`: Training and Evaluation Datasets

This folder holds the datasets that the FROST trainer and evaluator read. Apart from this file, its contents are **not committed** to git. The raw sources total about 7.6 GB and are distributed under their own licenses.

Every launcher runs with the **project root as its working directory** and passes
`--dataset_folder data/<dataset>`, so the paths below are relative to `<project root>`.

## Required layout

```text
data/
├── horse2zebra/
│   ├── train_A/          # domain A training images (horses)
│   ├── train_B/          # domain B training images (zebras)
│   ├── test_A/           # held-out domain A
│   ├── test_B/           # held-out domain B
│   ├── fixed_prompt_a.txt
│   ├── fixed_prompt_b.txt
│   └── masks/
│       ├── train_A/
│       ├── train_B/
│       ├── manifest.csv
│       └── manifest.json
└── bdd100k_day_night/
    └── (same structure)
```

## Expected image counts

| Dataset             | train_A | train_B | test_A | test_B |
| ------------------- | ------: | ------: | -----: | -----: |
| `horse2zebra`       |    1067 |    1334 |    120 |    140 |
| `bdd100k_day_night` |    1067 |    1334 |    120 |    140 |

`bdd100k_day_night` is a **horse-matched** Day→Night subset (seed 37), not the
balanced 1500/1500/500/500 subset described in the original brief. The raw BDD
images come from `100k/` + `labels/100k/`.

## Rebuilding `bdd100k_day_night`

The script's default paths point to the original training host, so pass **all four paths** yourself.
Write the regenerated manifests to a scratch folder, **not** to `manifests/bdd100k_day_night/`,
so the frozen manifests are never overwritten:

```bash
python scripts/build_bdd_day_night_subset.py \
  --image-root /path/to/bdd100k/100k \
  --label-root /path/to/bdd100k/labels/100k \
  --out-dir data/bdd100k_day_night \
  --manifest-dir /tmp/bdd_manifest_check \
  --seed 37 --train-day 1067 --train-night 1334 --val-day 120 --val-night 140
```

Then confirm the rebuild selected exactly the frozen images by comparing the four file lists
(their hashes are also recorded in `manifests/bdd100k_day_night/manifest_sha256.json`):

```bash
for f in train_day train_night val_day val_night; do
  cmp /tmp/bdd_manifest_check/$f.txt manifests/bdd100k_day_night/$f.txt && echo "$f OK"
done
```

`selection_config.json`, `weather_scene_distributions.json` and `AUDIT.md` record a build
timestamp (and, for two of them, local paths), so their hashes will differ on every rebuild; only
the four `.txt` lists must match.

`horse2zebra` uses the full standard split (1067/1334/120/140). The images are extracted from
`horse2zebra.zip`, and the archive folders `trainA/trainB/testA/testB` are renamed (or symlinked)
to `train_A/train_B/test_A/test_B`.

## Prompt files

Each prompt file contains exactly one non-empty line:

| File                                   | Contents                           |
| -------------------------------------- | ---------------------------------- |
| `horse2zebra/fixed_prompt_a.txt`       | `a horse`                          |
| `horse2zebra/fixed_prompt_b.txt`       | `a zebra`                          |
| `bdd100k_day_night/fixed_prompt_a.txt` | `a daytime street driving scene`   |
| `bdd100k_day_night/fixed_prompt_b.txt` | `a nighttime street driving scene` |

A→B uses `fixed_prompt_b`; B→A uses `fixed_prompt_a`.

## masks/

Precomputed granular masks exist for the **training splits only**. There are no test masks, because masks are used only in the training loss. Generate them with `bash scripts/precompute_masks.sh <dataset>`.

| Property          | Value                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------- |
| Files per image   | 3 mask modes + 1 stats file                                                                          |
| Per-dataset total | 2401 stems (1067 train_A + 1334 train_B)                                                             |
| Mask naming       | `<stem>_mask_{binary,fuzzy,fuzzy_rough}.npz`                                                         |
| Stats naming      | `<stem>_stats.npz` (keys `fine_L`, `fine_U`, `coarse_L`, `coarse_U`, `fine_mean_b`, `coarse_mean_b`) |
| Array             | `npz` with key `mask`, shape `286x286`, `float16`                                                    |
| Ball counts       | fine K=16 (8x8 patches), coarse K=4 (16x16 patches)                                                  |
| Seed              | 42                                                                                                   |
