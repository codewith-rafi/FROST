"""Tests for §10 Mag@4× structure-preservation loss (frost)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frost.structure_loss import structure_preservation_loss  # noqa: E402


def test_structure_loss_shape_and_nonneg():
    b, c, h, w = 1, 3, 64, 64
    x = torch.randn(b, c, h, w)
    g = x + 0.1 * torch.randn_like(x)
    m = torch.ones(b, 1, h, w)
    loss = structure_preservation_loss(x, g, m)
    assert loss.ndim == 0
    assert float(loss.item()) >= 0.0


def test_zero_mask_reduces_loss():
    x = torch.randn(1, 3, 64, 64)
    g = torch.randn(1, 3, 64, 64)
    m1 = torch.ones(1, 1, 64, 64)
    m0 = torch.zeros(1, 1, 64, 64)
    l1 = structure_preservation_loss(x, g, m1)
    l0 = structure_preservation_loss(x, g, m0)
    assert float(l0.item()) <= float(l1.item()) + 1e-6


def test_identical_images_near_zero_with_mask():
    x = torch.randn(1, 3, 64, 64)
    m = torch.ones(1, 1, 64, 64)
    loss = structure_preservation_loss(x, x.clone(), m)
    assert float(loss.item()) < 1e-5
