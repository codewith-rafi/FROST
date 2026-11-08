#!/usr/bin/env python3
"""Mask sanity diagnostics (instruction §15.4).

Basic stats (all modes) plus fuzzy-rough lower/upper approximation region fractions
derived from the sibling `<stem>_stats.npz` produced by the mask precompute step.

Usage:
  python scripts/mask_diagnostics.py --mask_dir <data>/<ds>/masks --mode fuzzy_rough \
      --out report/tables/mask_diagnostics/<ds>_fuzzy_rough.json
  python scripts/mask_diagnostics.py --mask_dir ... --mode binary --out ...
  python scripts/mask_diagnostics.py --all --masks_root <data_root> --out_dir report/tables/mask_diagnostics

Operationalization note (also written into every JSON): the stored artifacts keep
ball-level L/U (per ball, K=16 fine / K=4 coarse) and the final 286x286 mask, but not
per-pixel membership a_p nor ball assignment maps. Region fractions are therefore
measured as mask mass against ball-derived scalar thresholds, i.e. an approximation of
the exact lower/upper approximation membership sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# §7.5 scale weights used when combining fine/coarse masks
FINE_WEIGHT = 0.6
COARSE_WEIGHT = 0.4
OPERATIONALIZATION = (
    "Region fractions are threshold-based approximations: stored artifacts keep ball-level "
    "L/U and the final mask, not per-pixel membership a_p or ball assignment maps. "
    "L and U are computed per image as 0.6*mean(fine) + 0.4*mean(coarse) from stats.npz, "
    "matching the §7.5 scale weights."
)


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npz":
        data = np.load(path)
        if "mask" in data:
            return data["mask"].astype(np.float32)
        if "M" in data:
            return data["M"].astype(np.float32)
        return data[data.files[0]].astype(np.float32)
    if path.suffix == ".pt":
        import torch

        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, dict):
            t = obj.get("mask", obj.get("M"))
            return t.detach().float().cpu().numpy()
        return obj.detach().float().cpu().numpy()
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    raise ValueError(f"unsupported mask file: {path}")


def _stats_path(mask_path: Path) -> Path:
    """`<stem>_mask_<mode>.npz` -> `<stem>_stats.npz`."""
    stem = mask_path.stem
    marker = "_mask_"
    if marker in stem:
        stem = stem.split(marker, 1)[0]
    return mask_path.with_name(f"{stem}_stats.npz")


def _bounds(stats_path: Path) -> tuple[float, float] | None:
    if not stats_path.is_file():
        return None
    z = np.load(stats_path)
    need = ("fine_L", "fine_U", "coarse_L", "coarse_U")
    if not all(k in z.files for k in need):
        return None
    lo = FINE_WEIGHT * float(np.mean(z["fine_L"])) + COARSE_WEIGHT * float(np.mean(z["coarse_L"]))
    hi = FINE_WEIGHT * float(np.mean(z["fine_U"])) + COARSE_WEIGHT * float(np.mean(z["coarse_U"]))
    return lo, hi


def summarize(masks: list[np.ndarray], mode: str, bounds: list[tuple[float, float]]) -> dict:
    flat = np.concatenate([m.reshape(-1) for m in masks]).astype(np.float64)
    out = {
        "n_masks": len(masks),
        "n_pixels": int(flat.size),
        "mean_mask_value": float(flat.mean()),
        "p10": float(np.percentile(flat, 10)),
        "p50": float(np.percentile(flat, 50)),
        "p90": float(np.percentile(flat, 90)),
        "frac_gt_0_75": float((flat > 0.75).mean()),
        "frac_mid": float(((flat > 0.25) & (flat <= 0.75)).mean()),
        "frac_le_0_25": float((flat <= 0.25).mean()),
        "mode": mode,
    }

    if mode == "fuzzy_rough" and bounds:
        lower = boundary = outside = 0.0
        total = 0
        widths = []
        for m, (lo, hi) in zip(masks, bounds):
            v = m.reshape(-1).astype(np.float64)
            total += v.size
            lower += float((v >= hi).sum())
            boundary += float(((v >= lo) & (v < hi)).sum())
            outside += float((v < lo).sum())
            widths.append(hi - lo)
        out.update(
            {
                "lower_approx_fraction": lower / total,
                "boundary_fraction": boundary / total,
                "outside_upper_fraction": outside / total,
                "mean_boundary_width": float(np.mean(widths)),
                "n_images_with_stats": len(widths),
                "frac_sum_check": (lower + boundary + outside) / total,
            }
        )
    out["operationalization"] = OPERATIONALIZATION
    return out


def find_masks(root: Path, mode: str, limit: int) -> list[Path]:
    patterns = [
        f"**/mask_{mode}.npz",
        f"**/mask_{mode}.pt",
        f"**/mask_{mode}.npy",
        f"**/*_{mode}.npz",
        f"**/*_{mode}.pt",
    ]
    paths: list[Path] = []
    for pat in patterns:
        paths.extend(sorted(root.glob(pat)))
    seen, uniq = set(), []
    for p in paths:
        r = p.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(p)
    if limit > 0:
        uniq = uniq[:limit]
    return uniq


def run_combo(mask_root: Path, mode: str, limit: int) -> dict:
    paths = find_masks(mask_root, mode, limit)
    if not paths:
        raise SystemExit(f"no masks found under {mask_root} for mode={mode}")
    masks = [_load_mask(p) for p in paths]
    bounds: list[tuple[float, float]] = []
    if mode == "fuzzy_rough":
        for p in paths:
            b = _bounds(_stats_path(p))
            bounds.append(b if b else (0.0, 0.0))
        missing = sum(1 for b in bounds if b == (0.0, 0.0))
        if missing:
            print(f"  WARNING: {missing}/{len(paths)} images had no usable stats.npz")
    report = summarize(masks, mode, bounds)
    report["mask_dir"] = str(mask_root)
    report["example_files"] = [str(p) for p in paths[:5]]
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_dir", type=str, default=None, help="single dataset masks dir")
    ap.add_argument("--mode", type=str, default="fuzzy", choices=["binary", "fuzzy", "fuzzy_rough"])
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--all", action="store_true", help="run 2 datasets x 3 modes")
    ap.add_argument("--masks_root", type=str, default=None, help="dir containing <dataset>/masks")
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    if args.all:
        if not args.masks_root or not args.out_dir:
            raise SystemExit("--all requires --masks_root and --out_dir")
        root = Path(args.masks_root)
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        combined = {}
        for ds in ("horse2zebra", "bdd100k_day_night"):
            for mode in ("binary", "fuzzy", "fuzzy_rough"):
                mask_root = root / ds / "masks"
                print(f"[{ds}/{mode}] scanning {mask_root}")
                rep = run_combo(mask_root, mode, args.limit)
                (out_dir / f"{ds}_{mode}.json").write_text(json.dumps(rep, indent=2) + "\n")
                combined[f"{ds}/{mode}"] = rep
                print(f"  n={rep['n_masks']} mean={rep['mean_mask_value']:.4f}")
        (out_dir / "all.json").write_text(json.dumps(combined, indent=2) + "\n")
        print(f"wrote {len(combined)} combos to {out_dir}")
        return

    if not args.mask_dir:
        raise SystemExit("--mask_dir is required unless --all")
    rep = run_combo(Path(args.mask_dir), args.mode, args.limit)
    print(json.dumps(rep, indent=2))
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(rep, indent=2) + "\n")


if __name__ == "__main__":
    main()
