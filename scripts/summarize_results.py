#!/usr/bin/env python3
"""§17 result tables + §16 paired bootstrap from the 8 committed metrics.json.

Read-only over the evaluation artifacts: never re-runs eval.

Outputs (under report/tables/):
  results_horse.md / .csv
  results_bdd.md / .csv
  results_combined.md           4 methods x 2 directions per dataset (16 entries)
  bootstrap_dino.json / .md     12 paired CIs (2 datasets x 2 directions x 3 methods)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

METHODS = ["baseline", "binary", "fuzzy", "fuzzy_rough"]
DATASETS = {
    "horse2zebra": ("output/fresh_20261024/horse2zebra", "results_horse", "Horse-to-Zebra"),
    "bdd100k_day_night": ("output/fresh_20261027/bdd100k_day_night", "results_bdd", "BDD100K Day-to-Night"),
}
# §17 verdicts relative to baseline
BETTER = "better overall"
STRUCT_TRADEOFF = "structure trade-off"
DIST_TRADEOFF = "distribution trade-off"
WORSE = "worse overall"
MIXED = "mixed"


def load(root: Path, dataset: str) -> dict[str, dict]:
    out = {}
    for m in METHODS:
        p = root / DATASETS[dataset][0] / m / "metrics.json"
        out[m] = json.loads(p.read_text())
    return out


def verdict_direction(base: dict, cur: dict, direction: str) -> str:
    """§17 4-way rule applied to ONE direction (FID and DINO, lower is better)."""
    dfid = cur[direction]["fid"] - base[direction]["fid"]
    ddino = cur[direction]["dino_struct_mean"] - base[direction]["dino_struct_mean"]
    fid_better, dino_better = dfid < 0, ddino < 0
    if fid_better and dino_better:
        return BETTER
    if dino_better and not fid_better:
        return STRUCT_TRADEOFF
    if fid_better and not dino_better:
        return DIST_TRADEOFF
    return WORSE


def verdict_overall(v_a2b: str, v_b2a: str) -> str:
    """Both directions agree -> that verdict; otherwise 'mixed' (directions disagree)."""
    return v_a2b if v_a2b == v_b2a else MIXED


def write_dataset_table(root: Path, out_dir: Path, dataset: str) -> list[dict]:
    _, prefix, title = DATASETS[dataset]
    data = load(root, dataset)
    base = data["baseline"]
    rows = []
    for m in METHODS:
        d = data[m]
        if m == "baseline":
            v_a = v_b = v_all = "-"
        else:
            v_a = verdict_direction(base, d, "a2b")
            v_b = verdict_direction(base, d, "b2a")
            v_all = verdict_overall(v_a, v_b)
        rows.append(
            {
                "method": m,
                "fid_a2b": d["a2b"]["fid"],
                "fid_b2a": d["b2a"]["fid"],
                "dino_a2b": d["a2b"]["dino_struct_mean"],
                "dino_a2b_std": d["a2b"]["dino_struct_std"],
                "dino_b2a": d["b2a"]["dino_struct_mean"],
                "dino_b2a_std": d["b2a"]["dino_struct_std"],
                "verdict_a2b": v_a,
                "verdict_b2a": v_b,
                "verdict_overall": v_all,
            }
        )

    md = [f"# {title} — FID and DINO-Struct (§17)", ""]
    md.append("Lower is better for both metrics. DINO-Struct shown as mean ± std over held-out images.")
    md.append("")
    md.append("| Method | A→B FID | A→B DINO | B→A FID | B→A DINO | verdict A→B | verdict B→A | overall |")
    md.append("|---|---:|---:|---:|---:|---|---|---|")
    for r in rows:
        md.append(
            f"| {r['method']} | {r['fid_a2b']:.4f} | {r['dino_a2b']:.5f} ± {r['dino_a2b_std']:.5f} | "
            f"{r['fid_b2a']:.4f} | {r['dino_b2a']:.5f} ± {r['dino_b2a_std']:.5f} | "
            f"{r['verdict_a2b']} | {r['verdict_b2a']} | {r['verdict_overall']} |"
        )
    md.append("")
    md.append("Verdict rules (§17), applied per direction: both metrics down = better overall; "
              "DINO down with FID up = structure trade-off; FID down with DINO up = distribution "
              "trade-off; both up = worse overall. `overall` reports the direction verdict when "
              "both directions agree, and `mixed` when they disagree — the two translation "
              "directions are not symmetric, so a single cross-direction label would be misleading.")
    md.append("")
    (out_dir / f"{prefix}.md").write_text("\n".join(md))

    with open(out_dir / f"{prefix}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in r.items()})
    return rows


def write_combined(all_rows: dict[str, list[dict]], out_dir: Path) -> None:
    md = ["# Combined result table (2 datasets x 4 methods x 2 directions = 16 entries)", ""]
    md.append("| Dataset | Method | Direction | FID | DINO-Struct |")
    md.append("|---|---|---|---:|---:|")
    for ds, rows in all_rows.items():
        for r in rows:
            md.append(f"| {ds} | {r['method']} | A→B | {r['fid_a2b']:.4f} | {r['dino_a2b']:.5f} |")
            md.append(f"| {ds} | {r['method']} | B→A | {r['fid_b2a']:.4f} | {r['dino_b2a']:.5f} |")
    md.append("")
    (out_dir / "results_combined.md").write_text("\n".join(md))


def bootstrap(root: Path, out_dir: Path, n_resamples: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []
    for ds in DATASETS:
        data = load(root, ds)
        base = data["baseline"]
        for direction in ("a2b", "b2a"):
            b = np.asarray(base[direction]["dino_struct_per_image"], dtype=np.float64)
            for m in METHODS[1:]:
                cur = np.asarray(data[m][direction]["dino_struct_per_image"], dtype=np.float64)
                if cur.shape != b.shape:
                    raise SystemExit(f"pairing mismatch {ds}/{m}/{direction}: {cur.shape} vs {b.shape}")
                d = cur - b
                idx = rng.integers(0, d.size, size=(n_resamples, d.size))
                means = d[idx].mean(axis=1)
                lo, hi = np.percentile(means, [2.5, 97.5])
                results.append(
                    {
                        "dataset": ds,
                        "direction": direction,
                        "method": m,
                        "n_pairs": int(d.size),
                        "mean_delta": float(d.mean()),
                        "ci95_low": float(lo),
                        "ci95_high": float(hi),
                        "ci_excludes_zero": bool(lo > 0 or hi < 0),
                        "n_resamples": int(n_resamples),
                        "seed": int(seed),
                    }
                )
    (out_dir / "bootstrap_dino.json").write_text(json.dumps(results, indent=2) + "\n")

    md = ["# §16 Paired bootstrap on DINO-Struct (single seed = 42)", ""]
    md.append(f"Per-image paired differences vs baseline, {n_resamples:,} resamples, "
              "95% percentile CI, `numpy.random.default_rng(42)`.")
    md.append("")
    md.append("Negative delta = lower (better) DINO-Struct than baseline.")
    md.append("")
    md.append("| Dataset | Direction | Method | n | mean Δ | 95% CI | excludes 0 |")
    md.append("|---|---|---|---:|---:|---|---|")
    for r in results:
        md.append(
            f"| {r['dataset']} | {r['direction'].upper()} | {r['method']} | {r['n_pairs']} | "
            f"{r['mean_delta']:+.6f} | [{r['ci95_low']:+.6f}, {r['ci95_high']:+.6f}] | "
            f"{'yes' if r['ci_excludes_zero'] else 'no'} |"
        )
    md.append("")
    md.append("Caveat: these are within-run sampling intervals on one seed (42). They quantify "
              "held-out image sampling variability only; they do not establish method significance "
              "across seeds, and FID is reported as a point estimate because it is a global "
              "distribution metric (§16).")
    md.append("")
    (out_dir / "bootstrap_dino.md").write_text("\n".join(md))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = args.root.resolve()
    out_dir = (args.out_dir or root / "report" / "tables").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = {}
    for ds in DATASETS:
        all_rows[ds] = write_dataset_table(root, out_dir, ds)
        print(f"[{ds}] wrote results table")
    write_combined(all_rows, out_dir)
    print("[combined] wrote 16-entry table")

    bt = bootstrap(root, out_dir, args.resamples, args.seed)
    print(f"[bootstrap] {len(bt)} CIs, {args.resamples} resamples")

    # self-check against known values
    checks = [
        ("horse2zebra", "fuzzy", "fid_a2b", 49.7487),
        ("horse2zebra", "fuzzy", "fid_b2a", 160.2574),
        ("bdd100k_day_night", "fuzzy", "fid_a2b", 91.0697),
        ("bdd100k_day_night", "fuzzy", "fid_b2a", 124.1868),
    ]
    ok = True
    for ds, m, key, expect in checks:
        got = next(r for r in all_rows[ds] if r["method"] == m)[key]
        match = abs(got - expect) < 5e-4
        ok &= match
        print(f"  check {ds}/{m}/{key}: {got:.4f} vs {expect:.4f} -> {'OK' if match else 'MISMATCH'}")
    if not ok:
        print("SELF-CHECK FAILED", file=sys.stderr)
        return 1
    print("ALL RESULT GATES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
