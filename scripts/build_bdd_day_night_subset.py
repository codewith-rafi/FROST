#!/usr/bin/env python3
"""Deterministic stratified BDD100K day/night subset (instruction.md §4.2).

Within each (weather, scene) stratum, order by sha256(f\"{seed}:{basename}\").
Official train → train_*; official val → test_* (trainer names). Ignore test/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


KEEP_TOD = {"daytime", "night"}
DROP_TOD = {"dawn/dusk", "undefined"}


def sha_key(seed: int, basename: str) -> str:
    return hashlib.sha256(f"{seed}:{basename}".encode("utf-8")).hexdigest()


def largest_remainder_quotas(counts: dict[tuple, int], target: int) -> dict[tuple, int]:
    """Proportional quotas that sum exactly to target."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("empty pool")
    if target > total:
        raise ValueError(f"target {target} > pool {total}")
    raw = {k: (counts[k] * target) / total for k in counts}
    floors = {k: int(v) for k, v in raw.items()}
    rem = target - sum(floors.values())
    frac = sorted(((raw[k] - floors[k], k) for k in counts), reverse=True)
    quotas = dict(floors)
    for i in range(rem):
        quotas[frac[i][1]] += 1
    # Cap by available; leftover filled later from global remainder
    for k in list(quotas):
        if quotas[k] > counts[k]:
            quotas[k] = counts[k]
    return quotas


def load_pool(image_dir: Path, label_dir: Path, domain: str) -> list[dict]:
    """Load images for one official split filtered to one timeofday domain."""
    imgs = {p.stem: p for p in image_dir.glob("*.jpg")}
    rows = []
    seen = set()
    for jp in label_dir.glob("*.json"):
        data = json.loads(jp.read_text())
        attrs = data.get("attributes") or {}
        tod = attrs.get("timeofday")
        if tod not in KEEP_TOD:
            continue
        if tod != domain:
            continue
        stem = jp.stem
        if stem in seen:
            continue
        seen.add(stem)
        if stem not in imgs:
            continue
        basename = f"{stem}.jpg"
        rows.append(
            {
                "basename": basename,
                "stem": stem,
                "weather": attrs.get("weather"),
                "scene": attrs.get("scene"),
                "timeofday": tod,
                "image_path": imgs[stem],
                "label_path": jp,
            }
        )
    return rows


def select_stratified(rows: list[dict], target: int, seed: int) -> tuple[list[dict], dict]:
    by_group: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[(r["weather"], r["scene"])].append(r)

    for g in by_group:
        by_group[g].sort(key=lambda r: (sha_key(seed, r["basename"]), r["basename"]))

    counts = {g: len(v) for g, v in by_group.items()}
    quotas = largest_remainder_quotas(counts, target)

    selected: list[dict] = []
    selected_names: set[str] = set()
    shortfall = 0
    for g, q in quotas.items():
        take = by_group[g][:q]
        for r in take:
            selected.append(r)
            selected_names.add(r["basename"])
        if len(take) < q:
            shortfall += q - len(take)

    if len(selected) < target:
        need = target - len(selected)
        remainder = [r for r in rows if r["basename"] not in selected_names]
        remainder.sort(key=lambda r: (sha_key(seed, r["basename"]), r["basename"]))
        for r in remainder[:need]:
            selected.append(r)
            selected_names.add(r["basename"])
        shortfall_filled = need
    else:
        shortfall_filled = 0

    # Trim if over (should not happen)
    if len(selected) > target:
        selected = selected[:target]

    meta = {
        "pool_size": len(rows),
        "target": target,
        "selected": len(selected),
        "n_groups": len(by_group),
        "quotas": {f"{w}|{s}": q for (w, s), q in sorted(quotas.items(), key=lambda x: str(x[0]))},
        "group_sizes": {f"{w}|{s}": counts[(w, s)] for (w, s) in sorted(counts)},
        "shortfall_before_global_fill": shortfall,
        "global_fill_used": shortfall_filled,
    }
    # Stable selection order: hash then basename
    selected.sort(key=lambda r: (sha_key(seed, r["basename"]), r["basename"]))
    return selected, meta


def dist(rows: list[dict]) -> dict:
    c = Counter((r["weather"], r["scene"]) for r in rows)
    return {f"{w}|{s}": n for (w, s), n in sorted(c.items(), key=lambda x: (-x[1], str(x[0])))}


def tod_dist(rows: list[dict]) -> dict:
    return dict(Counter(r["timeofday"] for r in rows))


def materialize(selected: list[dict], out_dir: Path, link_type: str = "symlink") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # clear existing links/files
    for p in out_dir.iterdir():
        if p.is_symlink() or p.is_file():
            p.unlink()
    for r in selected:
        dst = out_dir / r["basename"]
        src = r["image_path"].resolve()
        if link_type == "symlink":
            dst.symlink_to(src)
        elif link_type == "hardlink":
            os.link(src, dst)
        else:
            raise ValueError(link_type)


def rgb_check(selected: list[dict]) -> list[dict]:
    bad = []
    for r in selected:
        try:
            with Image.open(r["image_path"]) as im:
                im.convert("RGB")
        except Exception as e:
            bad.append({"basename": r["basename"], "error": str(e)})
    return bad


def write_list(path: Path, selected: list[dict]) -> None:
    path.write_text("".join(r["basename"] + "\n" for r in selected))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-root", type=Path, default=Path("/workspace/rafi/frost/100k"))
    ap.add_argument("--label-root", type=Path, default=Path("/workspace/rafi/frost/labels/100k"))
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/workspace/rafi/frost/data/bdd100k_day_night"),
    )
    ap.add_argument(
        "--manifest-dir",
        type=Path,
        default=Path("/workspace/rafi/frost/manifests/bdd100k_day_night"),
    )
    ap.add_argument("--seed", type=int, default=37)
    ap.add_argument("--train-day", type=int, default=1067)
    ap.add_argument("--train-night", type=int, default=1334)
    ap.add_argument("--val-day", type=int, default=120)
    ap.add_argument("--val-night", type=int, default=140)
    ap.add_argument("--link-type", choices=["symlink", "hardlink"], default="symlink")
    args = ap.parse_args()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pools = {
        "train_day": load_pool(args.image_root / "train", args.label_root / "train", "daytime"),
        "train_night": load_pool(args.image_root / "train", args.label_root / "train", "night"),
        "val_day": load_pool(args.image_root / "val", args.label_root / "val", "daytime"),
        "val_night": load_pool(args.image_root / "val", args.label_root / "val", "night"),
    }
    targets = {
        "train_day": args.train_day,
        "train_night": args.train_night,
        "val_day": args.val_day,
        "val_night": args.val_night,
    }

    selected = {}
    select_meta = {}
    pre_dist = {}
    post_dist = {}
    for key, rows in pools.items():
        pre_dist[key] = {"timeofday": tod_dist(rows), "weather_scene": dist(rows), "n": len(rows)}
        sel, meta = select_stratified(rows, targets[key], args.seed)
        if len(sel) != targets[key]:
            raise SystemExit(f"{key}: got {len(sel)} want {targets[key]}")
        selected[key] = sel
        select_meta[key] = meta
        post_dist[key] = {"timeofday": tod_dist(sel), "weather_scene": dist(sel), "n": len(sel)}

    # Overlaps
    train_names = {r["basename"] for r in selected["train_day"] + selected["train_night"]}
    val_names = {r["basename"] for r in selected["val_day"] + selected["val_night"]}
    overlap = sorted(train_names & val_names)
    if overlap:
        raise SystemExit(f"train∩val overlap: {overlap[:10]} n={len(overlap)}")

    # Materialize trainer dirs
    mapping = {
        "train_day": "train_A",
        "train_night": "train_B",
        "val_day": "test_A",
        "val_night": "test_B",
    }
    for key, folder in mapping.items():
        materialize(selected[key], args.out_dir / folder, args.link_type)

    (args.out_dir / "fixed_prompt_a.txt").write_text("a daytime street driving scene\n")
    (args.out_dir / "fixed_prompt_b.txt").write_text("a nighttime street driving scene\n")

    # RGB check selected only
    bad = []
    for key in selected:
        bad.extend(rgb_check(selected[key]))
    if bad:
        raise SystemExit(f"RGB failures n={len(bad)} sample={bad[:5]}")

    # Manifests
    list_names = {
        "train_day": "train_day.txt",
        "train_night": "train_night.txt",
        "val_day": "val_day.txt",
        "val_night": "val_night.txt",
    }
    for key, fname in list_names.items():
        write_list(args.manifest_dir / fname, selected[key])

    config = {
        "frozen_at_utc": now,
        "seed": args.seed,
        "hash_rule": 'sha256(f"{seed}:{basename}") with basename including .jpg',
        "quota_rule": "largest_remainder proportional by (weather, scene); shortfall from global hash-sorted remainder",
        "image_root": str(args.image_root),
        "label_root": str(args.label_root),
        "out_dir": str(args.out_dir),
        "link_type": args.link_type,
        "targets": targets,
        "mapping": mapping,
        "keep_timeofday": sorted(KEEP_TOD),
        "drop_timeofday": sorted(DROP_TOD),
        "ignore_official_test": True,
        "training_seed_note": "instruction.md primary training seed remains 42; this seed is subset selection only",
        "prompts": {
            "fixed_prompt_a.txt": "a daytime street driving scene",
            "fixed_prompt_b.txt": "a nighttime street driving scene",
        },
        "select_meta": select_meta,
        "train_val_basename_overlap": 0,
        "rgb_open_failures": 0,
        "counts": {k: len(selected[k]) for k in selected},
    }
    (args.manifest_dir / "selection_config.json").write_text(json.dumps(config, indent=2) + "\n")

    distributions = {
        "frozen_at_utc": now,
        "pre_selection": pre_dist,
        "post_selection": post_dist,
    }
    (args.manifest_dir / "weather_scene_distributions.json").write_text(
        json.dumps(distributions, indent=2) + "\n"
    )

    audit = f"""# BDD100K day/night subset audit

- Frozen at UTC: {now}
- Subset seed (hash salt): {args.seed}
- Hash: `sha256(f"{{seed}}:{{basename}}")`
- Image root: `{args.image_root}`
- Label root: `{args.label_root}`
- Trainer root: `{args.out_dir}`
- Link type: {args.link_type}

## Counts

| Manifest | Official split | Domain | N |
|---|---|---|---:|
| train_day.txt | train | daytime | {len(selected['train_day'])} |
| train_night.txt | train | night | {len(selected['train_night'])} |
| val_day.txt | val | daytime | {len(selected['val_day'])} |
| val_night.txt | val | night | {len(selected['val_night'])} |

## Checks

- train∩val basename overlap: 0
- RGB open failures: 0
- Official test split: ignored for this subset
- Training seed (instruction §19): 42 (unchanged by this freeze)

## Prompts

- fixed_prompt_a.txt: a daytime street driving scene
- fixed_prompt_b.txt: a nighttime street driving scene
"""
    (args.manifest_dir / "AUDIT.md").write_text(audit)

    hashes = {}
    for name in list(list_names.values()) + [
        "selection_config.json",
        "weather_scene_distributions.json",
        "AUDIT.md",
    ]:
        hashes[name] = file_sha256(args.manifest_dir / name)
    (args.manifest_dir / "manifest_sha256.json").write_text(json.dumps(hashes, indent=2) + "\n")

    print("BDD_DAY_NIGHT_FREEZE_OK")
    print(json.dumps(config["counts"], indent=2))
    print("manifest_dir", args.manifest_dir)
    print("out_dir", args.out_dir)


if __name__ == "__main__":
    main()
