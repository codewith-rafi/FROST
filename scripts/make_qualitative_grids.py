#!/usr/bin/env python3
"""§15.3 qualitative comparison grids.

Stage 1 (--select): pre-register 6 stems per (dataset, direction) by
sha256("42:<filename>") over the sorted test list, BEFORE any output is viewed.
Stage 2 (--render): build 4 seven-panel grids from eval_final/ + masks_heldout/.

Panel order (§15.3): input, baseline, binary, fuzzy, fuzzy_rough, real target,
fuzzy-rough mask.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from glob import glob
from pathlib import Path

import numpy as np

METHODS = ["baseline", "binary", "fuzzy", "fuzzy_rough"]
DATASETS = {
    "horse2zebra": {
        "run_root": "output/fresh_20261024/horse2zebra",
        "data": "data/horse2zebra",
        "masks": "report/masks_heldout/horse2zebra",
        "fig_prefix": "qual_horse",
        "label": "Horse-to-Zebra",
    },
    "bdd100k_day_night": {
        "run_root": "output/fresh_20261027/bdd100k_day_night",
        "data": "data/bdd100k_day_night",
        "masks": "report/masks_heldout/bdd100k_day_night",
        "fig_prefix": "qual_bdd",
        "label": "BDD100K Day-to-Night",
    },
}
N_PER_DIRECTION = 6


def list_images(folder: Path) -> list[Path]:
    """Must match frost/evaluate_unpaired.py::_list_images exactly."""
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob(os.path.join(folder, ext)))
    return sorted(Path(p) for p in paths)


def select(root: Path) -> dict:
    out = {}
    for ds, cfg in DATASETS.items():
        for direction, split in (("a2b", "test_A"), ("b2a", "test_B")):
            files = list_images(root / cfg["data"] / split)
            ranked = sorted(files, key=lambda p: hashlib.sha256(f"42:{p.name}".encode()).hexdigest())
            chosen = ranked[:N_PER_DIRECTION]
            out[f"{ds}/{direction}"] = {
                "split": split,
                "method": "sha256('42:<filename>') ascending, first N in the sorted test list",
                "n_total": len(files),
                "stems": [p.stem for p in chosen],
                "files": [p.name for p in chosen],
                "index_in_sorted_test_list": [files.index(p) for p in chosen],
            }
    return out


def stage_symlinks(root: Path, selection: dict) -> None:
    """Symlink the chosen test images into .cache/heldout_qual/<ds>/{train_A,train_B}.

    precompute_granular_masks only walks train_A/train_B, hence the staging names.
    """
    for ds, cfg in DATASETS.items():
        for direction, split in (("a2b", "test_A"), ("b2a", "test_B")):
            entry = selection[f"{ds}/{direction}"]
            stage_name = "train_A" if split == "test_A" else "train_B"
            staged = root / ".cache" / "heldout_qual" / ds / stage_name
            staged.mkdir(parents=True, exist_ok=True)
            for name in entry["files"]:
                src = root / cfg["data"] / split / name
                dst = staged / name
                if dst.exists() or dst.is_symlink():
                    dst.unlink()
                dst.symlink_to(src.resolve())


def render(root: Path, selection: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ds, cfg in DATASETS.items():
        run_root = root / cfg["run_root"]
        for direction in ("a2b", "b2a"):
            entry = selection[f"{ds}/{direction}"]
            stems = entry["stems"]
            idxs = entry["index_in_sorted_test_list"]
            stage_name = "train_A" if direction == "a2b" else "train_B"

            # probe panel geometry
            sample = Image.open(run_root / "baseline" / "eval_final" / f"samples_{direction}" / f"{idxs[0]}.png")
            w, h = sample.size
            aspect = h / w
            max_panel_w = 2.4
            fig_w = max_panel_w * 7
            fig_h = max_panel_w * aspect * 6

            fig, axes = plt.subplots(
                6, 7, figsize=(fig_w, fig_h), constrained_layout=True,
                gridspec_kw={"hspace": 0.06, "wspace": 0.04},
            )
            col_titles = ["input", "baseline", "binary", "fuzzy", "fuzzy-rough",
                          "real target\n(unpaired sample)", "fuzzy-rough mask"]

            for row, (stem, idx) in enumerate(zip(stems, idxs)):
                input_img = Image.open(root / cfg["data"] / entry["split"] / entry["files"][row]).convert("RGB")
                panels = [np.asarray(input_img)]
                for m in METHODS:
                    p = run_root / m / "eval_final" / f"samples_{direction}" / f"{idx}.png"
                    panels.append(np.asarray(Image.open(p).convert("RGB")))
                ref = run_root / "baseline" / "eval_final" / f"ref_{direction}" / f"{idx}.png"
                panels.append(np.asarray(Image.open(ref).convert("RGB")))
                mask_p = root / cfg["masks"] / stage_name / f"{stem}_mask_fuzzy_rough.npz"
                mask = np.load(mask_p)["mask"] if mask_p.is_file() else np.zeros((286, 286), np.float32)
                panels.append(None)  # mask drawn separately

                for col in range(6):
                    ax = axes[row][col]
                    ax.imshow(panels[col].astype(np.uint8) if col == 0 else panels[col])
                    ax.set_xticks([]); ax.set_yticks([])
                    if row == 0:
                        ax.set_title(col_titles[col], fontsize=8)
                    if col == 0:
                        ax.set_ylabel(stem, fontsize=6, rotation=0, ha="right", va="center", labelpad=18)

                axm = axes[row][6]
                im = axm.imshow(mask, cmap="magma", vmin=0, vmax=1, aspect="equal")
                axm.set_xticks([]); axm.set_yticks([])
                if row == 0:
                    axm.set_title(col_titles[6], fontsize=8)

            cbar = fig.colorbar(im, ax=axes[:, 6], fraction=0.02, pad=0.01)
            cbar.set_label("mask value", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

            fig.suptitle(
                f"{cfg['label']} — {direction.upper()} (6 pre-registered held-out images). "
                "Reference is an unpaired real sample; mask is 286x286 training-time geometry.",
                fontsize=9,
            )
            outp = out_dir / f"{cfg['fig_prefix']}_{direction}.png"
            fig.savefig(outp, dpi=150)
            plt.close(fig)
            written.append(outp)
            print(f"wrote {outp} ({outp.stat().st_size/1024**2:.2f} MiB, panel {w}x{h})")
    return written


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--stage-links", action="store_true")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--selection", type=Path, default=None)
    args = ap.parse_args()

    root = args.root.resolve()
    sel_path = args.selection or root / "report" / "tables" / "qualitative_selection.json"

    if args.select:
        selection = select(root)
        sel_path.parent.mkdir(parents=True, exist_ok=True)
        sel_path.write_text(json.dumps(selection, indent=2) + "\n")
        print(f"wrote {sel_path}")
        for k, v in selection.items():
            print(f"  {k}: {len(v['stems'])} stems from {v['n_total']} candidates")
        return 0

    selection = json.loads(sel_path.read_text())
    if args.stage_links:
        stage_symlinks(root, selection)
        print("staged symlinks under .cache/heldout_qual/")
    if args.render:
        render(root, selection, root / "report" / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
