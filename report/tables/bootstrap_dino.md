# §16 Paired bootstrap on DINO-Struct (single seed = 42)

Per-image paired differences vs baseline, 10,000 resamples, 95% percentile CI, `numpy.random.default_rng(42)`.

Negative delta = lower (better) DINO-Struct than baseline.

| Dataset | Direction | Method | n | mean Δ | 95% CI | excludes 0 |
|---|---|---|---:|---:|---|---|
| horse2zebra | A2B | binary | 120 | +0.001364 | [+0.000745, +0.002029] | yes |
| horse2zebra | A2B | fuzzy | 120 | +0.001436 | [+0.000872, +0.002103] | yes |
| horse2zebra | A2B | fuzzy_rough | 120 | +0.002048 | [+0.001382, +0.002804] | yes |
| horse2zebra | B2A | binary | 140 | -0.000413 | [-0.000534, -0.000293] | yes |
| horse2zebra | B2A | fuzzy | 140 | +0.000594 | [+0.000425, +0.000772] | yes |
| horse2zebra | B2A | fuzzy_rough | 140 | -0.000095 | [-0.000215, +0.000028] | no |
| bdd100k_day_night | A2B | binary | 120 | +0.003904 | [+0.003410, +0.004413] | yes |
| bdd100k_day_night | A2B | fuzzy | 120 | +0.001262 | [+0.000749, +0.001753] | yes |
| bdd100k_day_night | A2B | fuzzy_rough | 120 | +0.001104 | [+0.000727, +0.001457] | yes |
| bdd100k_day_night | B2A | binary | 140 | -0.003399 | [-0.004204, -0.002646] | yes |
| bdd100k_day_night | B2A | fuzzy | 140 | -0.007531 | [-0.008477, -0.006632] | yes |
| bdd100k_day_night | B2A | fuzzy_rough | 140 | -0.008380 | [-0.009425, -0.007392] | yes |

Caveat: these are within-run sampling intervals on one seed (42). They quantify held-out image sampling variability only; they do not establish method significance across seeds, and FID is reported as a point estimate because it is a global distribution metric (§16).
