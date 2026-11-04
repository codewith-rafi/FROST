# BDD100K 100k images raw audit

- Date (UTC): 2026-10-23T19:40:59Z
- Source zip: `/workspace/rafi/bdd100k_images_100k.zip`
- Raw root: `/workspace/rafi/frost/100k`
- Unzip status: `ALL_DONE`

## instruction.md §4 verify checklist

| Check | Spec | Observed | Status |
|---|---|---|---|
| Official splits | train/val/test | 70000/10000/20000 | Pass |
| Cross-split collisions | none | {'train_cap_val': 0, 'train_cap_test': 0, 'val_cap_test': 0} | Pass |
| RGB sample | RGB | [{'split': 'train', 'name': '96f784ed-a3069de4.jpg', 'mode': 'RGB', 'size': [1280, 720]}, {'split': 'val', 'name': 'bd204fcd-75ab6967.jpg', 'mode': 'RGB', 'size': [1280, 720]}, {'split': 'test', 'name': 'd91c308d-90965127.jpg', 'mode': 'RGB', 'size': [1280, 720]}] | Pass (sample) |
| Layout | images/100k + labels | `frost/100k` only; labels missing at audit | Partial |
| Day/night subset | 4k+4k / 1k+1k | not built | Blocked without labels |
| timeOfDay filter | daytime/night | needs train.json/val.json | Blocked |

## Notes

- Ignore `test/` for the day/night subset unless labels appear.
- Treat `/workspace/rafi/frost` as BDD100K_ROOT with images at `100k/` (no `images/` prefix).

## Resolution

The checklist above reflects the state at the time of the raw audit, before the labels were available. The labels were added afterwards under `labels/100k/`, and the day/night subset was built and frozen from them. See [`../bdd100k_day_night/AUDIT.md`](../bdd100k_day_night/AUDIT.md) for the final subset and its checks.
