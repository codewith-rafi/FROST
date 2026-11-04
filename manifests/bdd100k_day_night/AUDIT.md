# BDD100K day/night subset audit

- Frozen at UTC: 2026-10-24T09:50:46Z
- Subset seed (hash salt): 37
- Hash: `sha256(f"{seed}:{basename}")`
- Image root: `/workspace/rafi/frost/100k`
- Label root: `/workspace/rafi/frost/labels/100k`
- Trainer root: `/workspace/rafi/frost/data/bdd100k_day_night`
- Link type: symlink

## Counts

| Manifest | Official split | Domain | N |
|---|---|---|---:|
| train_day.txt | train | daytime | 1067 |
| train_night.txt | train | night | 1334 |
| val_day.txt | val | daytime | 120 |
| val_night.txt | val | night | 140 |

## Checks

- train∩val basename overlap: 0
- RGB open failures: 0
- Official test split: ignored for this subset
- Training seed (instruction §19): 42 (unchanged by this freeze)

## Prompts

- fixed_prompt_a.txt: a daytime street driving scene
- fixed_prompt_b.txt: a nighttime street driving scene
