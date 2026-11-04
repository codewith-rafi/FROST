#!/usr/bin/env python3
"""Offline granular mask precomputation (instruction §8) — parallel CPU workers."""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from frost.granular_masks import GranularMaskConfig, compute_mask_for_image


def _list_images(folder: str) -> list[str]:
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob(os.path.join(folder, ext)))
    return sorted(paths)


def _process_one(args: tuple) -> dict:
    path, out_split, seed, fine_balls, coarse_balls = args
    cfg = GranularMaskConfig(seed=seed, fine_balls=fine_balls, coarse_balls=coarse_balls)
    stem = Path(path).stem
    img = (
        np.asarray(
            Image.open(path).convert("RGB").resize((286, 286), Image.LANCZOS),
            dtype=np.float32,
        )
        / 255.0
    )
    derived = compute_mask_for_image(img, mode="all", cfg=cfg)
    out_split = Path(out_split)
    for mode in ("binary", "fuzzy", "fuzzy_rough"):
        m = derived[mode]
        np.savez_compressed(out_split / f"{stem}_mask_{mode}.npz", mask=m.astype(np.float16))
    fine = derived["fine_balls"]
    coarse = derived["coarse_balls"]
    np.savez_compressed(
        out_split / f"{stem}_stats.npz",
        fine_mean_b=fine["mean_b"].astype(np.float16),
        coarse_mean_b=coarse["mean_b"].astype(np.float16),
        fine_L=fine["L_b"].astype(np.float16),
        fine_U=fine["U_b"].astype(np.float16),
        coarse_L=coarse["L_b"].astype(np.float16),
        coarse_U=coarse["U_b"].astype(np.float16),
    )
    return {
        "split": out_split.name,
        "filename": Path(path).name,
        "stem": stem,
        "fine_ball_count": int(fine["n_balls"]),
        "coarse_ball_count": int(coarse["n_balls"]),
        "mean_mask_binary": float(derived["binary"].mean()),
        "mean_mask_fuzzy": float(derived["fuzzy"].mean()),
        "mean_mask_fuzzy_rough": float(derived["fuzzy_rough"].mean()),
        "min_mask_fuzzy": float(derived["fuzzy"].min()),
        "max_mask_fuzzy": float(derived["fuzzy"].max()),
        "fine_purity_mean": float(fine["mean_b"].mean()) if fine["mean_b"].size else 0.0,
        "coarse_purity_mean": float(coarse["mean_b"].mean()) if coarse["mean_b"].size else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_folder", type=str, required=True)
    ap.add_argument("--mask_dir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fine_balls", type=int, default=16)
    ap.add_argument("--coarse_balls", type=int, default=4)
    ap.add_argument("--granular_persistence_scale", type=float, default=4.0)
    ap.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 8) - 2)))
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    mask_root = Path(args.mask_dir)
    mask_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for split in ("train_A", "train_B"):
        folder = os.path.join(args.dataset_folder, split)
        imgs = _list_images(folder)
        if args.limit > 0:
            imgs = imgs[: args.limit]
        out_split = mask_root / split
        out_split.mkdir(parents=True, exist_ok=True)
        # skip already-written stems (resume-friendly after killed slow run)
        todo = []
        for path in imgs:
            stem = Path(path).stem
            done = all(
                (out_split / f"{stem}_mask_{mode}.npz").is_file()
                for mode in ("binary", "fuzzy", "fuzzy_rough")
            )
            if not done:
                todo.append((path, str(out_split), args.seed, args.fine_balls, args.coarse_balls))

        print(f"{split}: {len(imgs)} images, {len(todo)} to compute, workers={args.workers}")
        if not todo:
            continue
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_process_one, t) for t in todo]
            for fut in tqdm(as_completed(futs), total=len(futs), desc=split):
                rows.append(fut.result())

    # rebuild rows from disk for full manifest
    rows = []
    for split in ("train_A", "train_B"):
        out_split = mask_root / split
        for path in _list_images(os.path.join(args.dataset_folder, split)):
            stem = Path(path).stem
            mpath = out_split / f"{stem}_mask_fuzzy.npz"
            if not mpath.is_file():
                continue
            m = np.load(mpath)["mask"].astype(np.float32)
            mb = np.load(out_split / f"{stem}_mask_binary.npz")["mask"].astype(np.float32)
            mr = np.load(out_split / f"{stem}_mask_fuzzy_rough.npz")["mask"].astype(np.float32)
            rows.append(
                {
                    "split": split,
                    "filename": Path(path).name,
                    "stem": stem,
                    "mean_mask_binary": float(mb.mean()),
                    "mean_mask_fuzzy": float(m.mean()),
                    "mean_mask_fuzzy_rough": float(mr.mean()),
                    "min_mask_fuzzy": float(m.min()),
                    "max_mask_fuzzy": float(m.max()),
                }
            )

    man_csv = mask_root / "manifest.csv"
    if rows:
        with open(man_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (mask_root / "manifest.json").write_text(
        json.dumps({"n": len(rows), "seed": args.seed, "workers_default": args.workers}, indent=2)
    )
    print(f"Wrote {len(rows)} mask sets → {mask_root}")


if __name__ == "__main__":
    main()
