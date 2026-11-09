# BDD100K Day-to-Night — FID and DINO-Struct (§17)

Lower is better for both metrics. DINO-Struct shown as mean ± std over held-out images.

| Method | A→B FID | A→B DINO | B→A FID | B→A DINO | verdict A→B | verdict B→A | overall |
|---|---:|---:|---:|---:|---|---|---|
| baseline | 100.7398 | 0.01802 ± 0.00705 | 125.4937 | 0.03221 ± 0.01346 | - | - | - |
| binary | 96.4970 | 0.02193 ± 0.00923 | 127.3607 | 0.02881 ± 0.01156 | distribution trade-off | structure trade-off | mixed |
| fuzzy | 91.0697 | 0.01928 ± 0.00610 | 124.1868 | 0.02468 ± 0.01033 | distribution trade-off | better overall | mixed |
| fuzzy_rough | 100.9038 | 0.01913 ± 0.00663 | 126.0886 | 0.02383 ± 0.00960 | worse overall | structure trade-off | mixed |

Verdict rules (§17), applied per direction: both metrics down = better overall; DINO down with FID up = structure trade-off; FID down with DINO up = distribution trade-off; both up = worse overall. `overall` reports the direction verdict when both directions agree, and `mixed` when they disagree — the two translation directions are not symmetric, so a single cross-direction label would be misleading.
