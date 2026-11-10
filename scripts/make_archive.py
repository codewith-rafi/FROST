#!/usr/bin/env python3
"""Build a code+results archive of the frost project (no datasets, no checkpoints).

`zip`/`unzip` are not installed on this host, so this uses the stdlib `zipfile`.

Excludes (applied to repo-relative paths):
  - root data dirs: 100k/, labels/, horse2zebra/, data/
  - any directory named: checkpoints, __pycache__, wandb, .pytest_cache, .cache, .tmux, .git
  - any file ending: .pkl
  - any symlink (defence in depth: the tree has 2681 symlinks into datasets)

Usage:
  python scripts/make_archive.py --dry-run
  python scripts/make_archive.py --out /workspace/rafi/artifacts/frost_archive_20261102.zip
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

EXCLUDED_ROOT_DIRS = {"100k", "labels", "horse2zebra", "data"}
EXCLUDED_DIR_NAMES = {
    "checkpoints",
    "__pycache__",
    "wandb",
    ".pytest_cache",
    ".cache",
    ".tmux",
    ".git",
}
EXCLUDED_SUFFIXES = {".pkl"}


def is_excluded(rel: Path) -> bool:
    """True if this repo-relative path must not enter the archive."""
    parts = rel.parts
    if not parts:
        return False
    if parts[0] in EXCLUDED_ROOT_DIRS:
        return True
    if any(p in EXCLUDED_DIR_NAMES for p in parts):
        return True
    if rel.suffix in EXCLUDED_SUFFIXES:
        return True
    return False


def walk(root: Path):
    """Yield repo-relative file paths, pruning excluded directories before descending."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        rel_dir = Path(dirpath).relative_to(root)
        dirnames[:] = [d for d in dirnames if not is_excluded(rel_dir / d)]
        for name in filenames:
            rel = rel_dir / name
            if is_excluded(rel):
                continue
            abs_path = root / rel
            if abs_path.is_symlink():
                # Do not follow links out of the project tree.
                continue
            if not abs_path.is_file():
                continue
            yield rel, abs_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root: Path = args.root.resolve()
    if not (root / "src" / "frost").is_dir():
        raise SystemExit(f"not a frost project root: {root}")

    entries = list(walk(root))
    total = sum(p.stat().st_size for _, p in entries)

    by_top: dict[str, list[int]] = {}
    for rel, p in entries:
        top = rel.parts[0] if len(rel.parts) > 1 else "<root files>"
        acc = by_top.setdefault(top, [0, 0])
        acc[0] += p.stat().st_size
        acc[1] += 1

    print(f"root       : {root}")
    print(f"files      : {len(entries)}")
    print(f"size       : {total / 1e9:.2f} GB ({total / 1024**2:.0f} MiB)")
    print("\ntop-level breakdown:")
    for name, (size, count) in sorted(by_top.items(), key=lambda kv: -kv[1][0]):
        print(f"  {name:22s} {size / 1024**2:9.1f} MiB  ({count} files)")

    if args.dry_run:
        return 0

    if args.out is None:
        raise SystemExit("--out is required unless --dry-run")

    out: Path = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        out, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True, compresslevel=1
    ) as zf:
        for i, (rel, abs_path) in enumerate(entries, 1):
            zf.write(abs_path, arcname=rel.as_posix())
            if i % 500 == 0:
                print(f"  archived {i}/{len(entries)}", flush=True)

    print(f"\nwrote {out} ({out.stat().st_size / 1e9:.2f} GB)")

    # --- verification gates -------------------------------------------------
    problems: list[str] = []
    with zipfile.ZipFile(out) as zf:
        bad = zf.testzip()
        if bad is not None:
            problems.append(f"corrupt member: {bad}")
        names = zf.namelist()

    print(f"entries in archive: {len(names)}")
    if not (len(entries) - 50 <= len(names) <= len(entries) + 50):
        problems.append(f"entry count {len(names)} vs walked {len(entries)}")

    banned = (
        ("100k/", lambda n: n.startswith("100k/")),
        ("labels/", lambda n: n.startswith("labels/")),
        ("horse2zebra/", lambda n: n.startswith("horse2zebra/")),
        ("data/", lambda n: n.startswith("data/")),
        ("checkpoints/", lambda n: "/checkpoints/" in n),
        (".pkl", lambda n: n.endswith(".pkl")),
        (".git/", lambda n: "/.git/" in n),
        ("__pycache__", lambda n: "__pycache__" in n),
    )
    for label, pred in banned:
        hits = [n for n in names if pred(n)]
        print(f"  {label:16s} {len(hits)}")
        if hits:
            problems.append(f"{len(hits)} excluded-pattern entries ({label}), e.g. {hits[0]}")

    required = [
        ("metrics.json", 8),
        ("train.log", 8),
        ("eval.log", 8),
    ]
    for suffix, expect in required:
        n = sum(1 for x in names if x.endswith("/" + suffix) or x == suffix)
        print(f"  {suffix:16s} {n} (expect {expect})")
        if n < expect:
            problems.append(f"only {n} {suffix} (expected {expect})")

    must_have = [
        "src/frost/train_cyclegan_turbo.py",
        "scripts/run_course_eval.sh",
        "report/tables/vendor_patches/img2img-turbo-content.patch",
    ]
    for name in must_have:
        present = name in names
        print(f"  {'OK  ' if present else 'MISS'} {name}")
        if not present:
            problems.append(f"missing required entry: {name}")

    size_gb = out.stat().st_size / 1e9
    if not (1.6 <= size_gb <= 2.2):
        problems.append(f"archive size {size_gb:.2f} GB outside expected 1.6-2.2 GB")

    if problems:
        print("\nFAILED GATES:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nALL ARCHIVE GATES PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
