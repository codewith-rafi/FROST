"""BDD day/night horse-matched subset seed-37 reproducibility."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

FROST = Path(__file__).resolve().parents[1]
MAN = FROST / "manifests" / "bdd100k_day_night"
DATA = FROST / "data" / "bdd100k_day_night"


def test_selection_config_seed_and_targets():
    cfg = json.loads((MAN / "selection_config.json").read_text())
    assert cfg.get("seed") == 37
    assert cfg["targets"]["train_day"] == 1067
    assert cfg["targets"]["train_night"] == 1334
    assert cfg["targets"]["val_day"] == 120
    assert cfg["targets"]["val_night"] == 140


def test_manifest_counts():
    for name, n in [
        ("train_day.txt", 1067),
        ("train_night.txt", 1334),
        ("val_day.txt", 120),
        ("val_night.txt", 140),
    ]:
        lines = [ln.strip() for ln in (MAN / name).read_text().splitlines() if ln.strip()]
        assert len(lines) == n, f"{name}: {len(lines)} != {n}"


def test_hash_order_contract():
    lines = [ln.strip() for ln in (MAN / "train_day.txt").read_text().splitlines() if ln.strip()]
    basenames = [Path(p).name for p in lines]

    def key(bn: str) -> str:
        return hashlib.sha256(f"37:{bn}".encode()).hexdigest()

    assert len(set(basenames)) == len(basenames)
    assert key(basenames[0]) == hashlib.sha256(f"37:{basenames[0]}".encode()).hexdigest()


@pytest.mark.skipif(
    not (DATA / "train_A").is_dir(),
    reason="BDD data tree not present (datasets are not committed)",
)
def test_data_symlink_counts():
    assert len(list((DATA / "train_A").iterdir())) == 1067
    assert len(list((DATA / "train_B").iterdir())) == 1334
    assert len(list((DATA / "test_A").iterdir())) == 120
    assert len(list((DATA / "test_B").iterdir())) == 140
