"""Aligned image/mask crop-flip (frost.dataset)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frost.dataset import _aligned_train_pair  # noqa: E402


def test_aligned_shapes():
    random.seed(0)
    img = Image.fromarray(np.random.randint(0, 255, (300, 320, 3), dtype=np.uint8))
    mask = torch.rand(1, 286, 286)
    img_t, mask_t = _aligned_train_pair(img, mask)
    assert img_t.shape == (3, 256, 256)
    assert mask_t.shape == (1, 256, 256)
    assert float(mask_t.min()) >= 0.0
    assert float(mask_t.max()) <= 1.0


def test_unit_mask_when_none():
    random.seed(1)
    img = Image.fromarray(np.zeros((286, 286, 3), dtype=np.uint8))
    img_t, mask_t = _aligned_train_pair(img, None)
    assert torch.allclose(mask_t, torch.ones_like(mask_t))
