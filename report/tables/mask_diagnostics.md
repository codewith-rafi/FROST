# §15.4 Mask diagnostics (training masks only, no held-out tuning)

Computed over all training masks: `train_A` 1067 + `train_B` 1334 = 2401 images per dataset per mode (196,392,196 pixels).

## Horse-to-Zebra

| Mode | n | mean | p10 | p50 | p90 | frac>0.75 | frac 0.25-0.75 | frac<=0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| binary | 2401 | 0.4451 | 0.0000 | 0.3999 | 1.0000 | 0.3126 | 0.2771 | 0.4103 |
| fuzzy | 2401 | 0.2214 | 0.1133 | 0.2172 | 0.3372 | 0.0000 | 0.3580 | 0.6420 |
| fuzzy_rough | 2401 | 0.1114 | 0.0268 | 0.1017 | 0.2096 | 0.0000 | 0.0350 | 0.9650 |

Fuzzy-rough region diagnostics (against ball-level lower/upper approximations):

| Dataset | lower-approx frac | boundary frac | outside-upper frac | mean B_b | sum check |
|---|---:|---:|---:|---:|---:|
| Horse-to-Zebra | 0.0057 | 0.1572 | 0.8371 | 0.1098 | 1.0000 |

## BDD100K Day-to-Night

| Mode | n | mean | p10 | p50 | p90 | frac>0.75 | frac 0.25-0.75 | frac<=0.25 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| binary | 2401 | 0.3757 | 0.0000 | 0.3999 | 1.0000 | 0.2616 | 0.2464 | 0.4920 |
| fuzzy | 2401 | 0.1459 | 0.0532 | 0.1250 | 0.2754 | 0.0000 | 0.1402 | 0.8598 |
| fuzzy_rough | 2401 | 0.0570 | 0.0019 | 0.0372 | 0.1447 | 0.0000 | 0.0066 | 0.9934 |

Fuzzy-rough region diagnostics (against ball-level lower/upper approximations):

| Dataset | lower-approx frac | boundary frac | outside-upper frac | mean B_b | sum check |
|---|---:|---:|---:|---:|---:|
| BDD100K Day-to-Night | 0.0094 | 0.1074 | 0.8832 | 0.1000 | 1.0000 |

## Operationalization and caveats

- Region fractions are threshold-based: stored artifacts keep ball-level `L`/`U` and the final mask, but not per-pixel membership `a_p` or ball assignment maps. Per image, `L = 0.6*mean(fine_L) + 0.4*mean(coarse_L)` and `U = 0.6*mean(fine_U) + 0.4*mean(coarse_U)`, matching the §7.5 scale weights; the categories are `M >= U`, `L <= M < U`, `M < L`. The three fractions sum to 1 by construction (checked above).
- The binary masks are near-binary rather than strictly 0/1: §7.4 quantises per ball, but §7.5 then combines `0.6*fine + 0.4*coarse` and resizes 288->286 bilinearly, which reintroduces intermediate values.
- Diagnostics use training masks only; held-out masks were computed separately for the qualitative figures and never used to tune anything.
