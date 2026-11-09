# Horse-to-Zebra — FID and DINO-Struct (§17)

Lower is better for both metrics. DINO-Struct shown as mean ± std over held-out images.

| Method | A→B FID | A→B DINO | B→A FID | B→A DINO | verdict A→B | verdict B→A | overall |
|---|---:|---:|---:|---:|---|---|---|
| baseline | 56.0516 | 0.01862 ± 0.00812 | 163.3395 | 0.01704 ± 0.00602 | - | - | - |
| binary | 49.9145 | 0.01999 ± 0.00833 | 166.0005 | 0.01663 ± 0.00590 | distribution trade-off | structure trade-off | mixed |
| fuzzy | 49.7487 | 0.02006 ± 0.00878 | 160.2574 | 0.01763 ± 0.00624 | distribution trade-off | distribution trade-off | distribution trade-off |
| fuzzy_rough | 51.5349 | 0.02067 ± 0.00909 | 163.6094 | 0.01695 ± 0.00600 | distribution trade-off | structure trade-off | mixed |

Verdict rules (§17), applied per direction: both metrics down = better overall; DINO down with FID up = structure trade-off; FID down with DINO up = distribution trade-off; both up = worse overall. `overall` reports the direction verdict when both directions agree, and `mixed` when they disagree — the two translation directions are not symmetric, so a single cross-direction label would be misleading.
