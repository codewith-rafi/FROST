"""Tests for frost granular mask module."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frost.granular_masks import (  # noqa: E402
    GranularMaskConfig,
    compute_mask_for_image,
    compute_structural_statistics,
    derive_masks,
)


def _fake_rgb(h=286, w=286, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((h, w, 3), dtype=np.float64)


def test_mask_shapes_286():
    img = _fake_rgb()
    out = compute_mask_for_image(
        img, mode="all", cfg=GranularMaskConfig(seed=42, fine_balls=8, coarse_balls=4)
    )
    for mode in ("binary", "fuzzy", "fuzzy_rough"):
        assert out[mode].shape == (286, 286)
        assert out[mode].dtype == np.float32
        assert float(out[mode].min()) >= 0.0
        assert float(out[mode].max()) <= 1.0 + 1e-5


def test_modes_differ_on_structure():
    img = np.zeros((286, 286, 3), dtype=np.float64)
    img[:, :143, :] = 1.0
    out = compute_mask_for_image(
        img, mode="all", cfg=GranularMaskConfig(seed=42, fine_balls=8, coarse_balls=4)
    )
    assert out["fuzzy"].mean() > 0.0

    # §7.4 defines the binary mask as 0/1 at ball level, but §7.5 then combines the two
    # scales as 0.6*fine + 0.4*coarse and bilinearly resizes 288->286. The written
    # "binary" mask is therefore *near*-binary, not strictly {0, 1}: measured on the real
    # precomputed masks ~72% of pixels are exactly 0 or 1 (291-320 distinct values), and
    # ~90% on this synthetic split. Training used these masks, so guard the quantisation
    # that actually holds instead of an unattainable exact set.
    binary = out["binary"]
    hard = float(((binary == 0.0) | (binary == 1.0)).mean())
    assert hard > 0.85
    # must carry structure, never degenerate to all-zero or all-one
    assert 0.0 < float(binary.mean()) < 1.0


def test_statistics_pad_288():
    stats = compute_structural_statistics(_fake_rgb(), GranularMaskConfig())
    assert stats["Y"].shape == (288, 288)
    assert stats["a_map"].shape == (288, 288)
    d = derive_masks(stats, mode="fuzzy")
    assert d["fuzzy"].shape == (286, 286)
