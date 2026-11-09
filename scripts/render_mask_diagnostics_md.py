#!/usr/bin/env python3
"""Render the combined §15.4 mask diagnostics markdown from the per-combo JSONs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = ["horse2zebra", "bdd100k_day_night"]
MODES = ["binary", "fuzzy", "fuzzy_rough"]
TITLES = {"horse2zebra": "Horse-to-Zebra", "bdd100k_day_night": "BDD100K Day-to-Night"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("report/tables/mask_diagnostics"))
    ap.add_argument("--out", type=Path, default=Path("report/tables/mask_diagnostics.md"))
    args = ap.parse_args()

    data = {}
    for ds in DATASETS:
        for mode in MODES:
            p = args.dir / f"{ds}_{mode}.json"
            if not p.is_file():
                raise SystemExit(f"missing {p}")
            data[(ds, mode)] = json.loads(p.read_text())

    md = ["# §15.4 Mask diagnostics (training masks only, no held-out tuning)", ""]
    md.append("Computed over all training masks: `train_A` 1067 + `train_B` 1334 = 2401 images per "
              "dataset per mode (196,392,196 pixels).")
    md.append("")

    for ds in DATASETS:
        md.append(f"## {TITLES[ds]}")
        md.append("")
        md.append("| Mode | n | mean | p10 | p50 | p90 | frac>0.75 | frac 0.25-0.75 | frac<=0.25 |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for mode in MODES:
            r = data[(ds, mode)]
            md.append(
                f"| {mode} | {r['n_masks']} | {r['mean_mask_value']:.4f} | {r['p10']:.4f} | "
                f"{r['p50']:.4f} | {r['p90']:.4f} | {r['frac_gt_0_75']:.4f} | "
                f"{r['frac_mid']:.4f} | {r['frac_le_0_25']:.4f} |"
            )
        md.append("")
        md.append("Fuzzy-rough region diagnostics (against ball-level lower/upper approximations):")
        md.append("")
        md.append("| Dataset | lower-approx frac | boundary frac | outside-upper frac | mean B_b | sum check |")
        md.append("|---|---:|---:|---:|---:|---:|")
        r = data[(ds, "fuzzy_rough")]
        md.append(
            f"| {TITLES[ds]} | {r['lower_approx_fraction']:.4f} | {r['boundary_fraction']:.4f} | "
            f"{r['outside_upper_fraction']:.4f} | {r['mean_boundary_width']:.4f} | "
            f"{r['frac_sum_check']:.4f} |"
        )
        md.append("")

    md.append("## Operationalization and caveats")
    md.append("")
    md.append("- Region fractions are threshold-based: stored artifacts keep ball-level `L`/`U` and the "
              "final mask, but not per-pixel membership `a_p` or ball assignment maps. Per image, "
              "`L = 0.6*mean(fine_L) + 0.4*mean(coarse_L)` and `U = 0.6*mean(fine_U) + 0.4*mean(coarse_U)`, "
              "matching the §7.5 scale weights; the categories are `M >= U`, `L <= M < U`, `M < L`. "
              "The three fractions sum to 1 by construction (checked above).")
    md.append("- The binary masks are near-binary rather than strictly 0/1: §7.4 quantises per ball, but "
              "§7.5 then combines `0.6*fine + 0.4*coarse` and resizes 288->286 bilinearly, which "
              "reintroduces intermediate values.")
    md.append("- Diagnostics use training masks only; held-out masks were computed separately for the "
              "qualitative figures and never used to tune anything.")
    md.append("")

    args.out.write_text("\n".join(md))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
