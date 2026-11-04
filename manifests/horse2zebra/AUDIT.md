# Horse-to-Zebra raw audit

- Date (UTC): 2026-10-23T19:40:59Z
- Source zip: `/workspace/rafi/horse2zebra.zip`
- Raw root: `/workspace/rafi/frost/horse2zebra`
- Unzip status: `ALL_DONE` (`/workspace/rafi/logs/unzip_frost_status.txt`)

## instruction.md §4 verify checklist

| Check | Spec | Observed | Status |
|---|---|---|---|
| Train horses | 1,067 | 1067 | Pass |
| Train zebras | 1,334 | 1334 | Pass |
| Test horses | 120 | 120 | Pass |
| Test zebras | 140 | 140 | Pass |
| Filename overlaps | none | {'trainA_cap_testA': 0, 'trainB_cap_testB': 0, 'trainA_cap_trainB': 0, 'testA_cap_testB': 0} | Pass |
| Subsample | use all | full set | Pass |
| Prompt files | fixed_prompt_a/b.txt | absent in raw tree | Fail until freeze |
| Trainer folder names | train_A/... | still trainA/... | Fail until freeze |
| RGB open | every image | sample pass; full pass at freeze | Partial |

## Notes

- Horse-to-Zebra is used in full, without subsampling.
- The raw tree is read-only after the freeze; the trainer view lives under `data/horse2zebra/`.
- The two items marked "Fail until freeze" were resolved by the freeze below.

## Freeze (trainer view)

- Frozen at UTC: 2026-10-23T19:42:15Z
- Trainer root: `/workspace/rafi/frost/data/horse2zebra`
- Link type: symlink from archive dirs (`trainA`→`train_A`, etc.)
- Prompts: `a horse` / `a zebra` (one line each)
- Full RGB open pass: 2,661 / 2,661 OK
- Manifests: `train_A.txt` … `test_B.txt`, `freeze.json`
- Raw tree `/workspace/rafi/frost/horse2zebra` left unchanged
