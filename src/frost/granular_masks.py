"""Granular mask construction for CycleGAN-Turbo (instruction §7 / §12.1).

Streamlined Fixed-K MiniBatchKMeans pipeline:
  luminance → signed-gradient persistence → fine/coarse balls → binary/fuzzy/fuzzy_rough
Masks are computed on a 288×288 padded grid and returned at 286×286.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


EPS = 1e-6


def _fixed_kmeans(X: np.ndarray, k: int, seed: int, max_iter: int = 25) -> np.ndarray:
    """Deterministic Lloyd Fixed-K (replaces slow sklearn MiniBatchKMeans for offline masks)."""
    n, d = X.shape
    k = int(min(max(k, 1), n))
    rng = np.random.default_rng(seed)
    # k-means++ style init (lightweight)
    cents = np.empty((k, d), dtype=np.float64)
    first = int(rng.integers(0, n))
    cents[0] = X[first]
    closest = np.full(n, np.inf)
    for j in range(1, k):
        dist = ((X - cents[j - 1]) ** 2).sum(axis=1)
        closest = np.minimum(closest, dist)
        probs = closest / (closest.sum() + EPS)
        cents[j] = X[int(rng.choice(n, p=probs))]
    labels = np.zeros(n, dtype=np.int32)
    for _ in range(max_iter):
        # (n,k) squared distances
        d2 = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)
        new_labels = d2.argmin(axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for b in range(k):
            m = labels == b
            if np.any(m):
                cents[b] = X[m].mean(axis=0)
            else:
                cents[b] = X[int(rng.integers(0, n))]
    return labels


@dataclass
class GranularMaskConfig:
    fine_patch_size: int = 8
    coarse_patch_size: int = 16
    fine_balls: int = 16
    coarse_balls: int = 4
    max_fine_balls: int = 64
    max_coarse_balls: int = 16
    binary_threshold: float = 0.50
    lower_percentile: float = 10.0
    upper_percentile: float = 90.0
    boundary_penalty: float = 0.25
    purity_threshold: float = 0.80
    fine_weight: float = 0.60
    coarse_weight: float = 0.40
    seed: int = 42
    target_size: int = 286
    pad_size: int = 288


def rgb_to_luminance(rgb: np.ndarray) -> np.ndarray:
    """rgb in [0,1], shape (H,W,3) → Y (H,W)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _pad_to(y: np.ndarray, size: int) -> np.ndarray:
    h, w = y.shape
    if h == size and w == size:
        return y
    out = np.zeros((size, size), dtype=y.dtype)
    out[:h, :w] = y
    # replicate edge into remainder
    if w < size:
        out[:h, w:] = y[:, -1:]
    if h < size:
        out[h:, :] = out[h - 1 : h, :]
    return out


def _sobel(y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sobel gradients with reflect pad (vectorized)."""
    ypad = np.pad(y, 1, mode="reflect")
    # separable-ish 3x3 sobel via slicing
    yx = (
        -ypad[0:-2, 0:-2] + ypad[0:-2, 2:]
        - 2 * ypad[1:-1, 0:-2] + 2 * ypad[1:-1, 2:]
        - ypad[2:, 0:-2] + ypad[2:, 2:]
    )
    yy = (
        -ypad[0:-2, 0:-2] - 2 * ypad[0:-2, 1:-1] - ypad[0:-2, 2:]
        + ypad[2:, 0:-2] + 2 * ypad[2:, 1:-1] + ypad[2:, 2:]
    )
    return yx.astype(np.float64), yy.astype(np.float64)


def _avg_pool(arr: np.ndarray, k: int) -> np.ndarray:
    h, w = arr.shape
    nh, nw = h // k, w // k
    truncated = arr[: nh * k, : nw * k]
    return truncated.reshape(nh, k, nw, k).mean(axis=(1, 3))


def _upsample_nn(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest upsample to out size."""
    yh = np.linspace(0, arr.shape[0] - 1, out_h)
    xw = np.linspace(0, arr.shape[1] - 1, out_w)
    yi = np.clip(np.round(yh).astype(int), 0, arr.shape[0] - 1)
    xi = np.clip(np.round(xw).astype(int), 0, arr.shape[1] - 1)
    return arr[yi][:, xi]


def _resize_bilinear(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Bilinear resize via numpy mesh sampling."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float64)
    ys = np.linspace(0, h - 1, out_h, dtype=np.float64)
    xs = np.linspace(0, w - 1, out_w, dtype=np.float64)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    wy = (ys - y0)[:, None]
    wx = (xs - x0)[None, :]
    Ia = arr[y0][:, x0]
    Ib = arr[y0][:, x1]
    Ic = arr[y1][:, x0]
    Id = arr[y1][:, x1]
    return (Ia * (1 - wy) * (1 - wx) + Ib * (1 - wy) * wx + Ic * wy * (1 - wx) + Id * wy * wx)


def compute_structural_statistics(
    image: np.ndarray,
    cfg: Optional[GranularMaskConfig] = None,
) -> Dict:
    """Compute padded luminance maps and per-pixel / ready-for-patch membership a.

    `image`: RGB float [0,1] or uint8, any HxW (will be resized conceptually by caller to 286).
    Returns dict with Y_pad (288), a_map (288), g1, g8_up.
    """
    cfg = cfg or GranularMaskConfig()
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        # CHW → HWC
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        y = image.astype(np.float64)
        if y.max() > 1.5:
            y = y / 255.0
    else:
        y = rgb_to_luminance(image)

    # ensure 286 then pad 288
    if y.shape[0] != cfg.target_size or y.shape[1] != cfg.target_size:
        y = _resize_bilinear(y, cfg.target_size, cfg.target_size)
    y_pad = _pad_to(y, cfg.pad_size)

    yx, yy = _sobel(y_pad)
    g1 = np.sqrt(yx ** 2 + yy ** 2)
    yx8 = _avg_pool(yx, 8)
    yy8 = _avg_pool(yy, 8)
    g8 = np.sqrt(yx8 ** 2 + yy8 ** 2)
    g8_up = _upsample_nn(g8, cfg.pad_size, cfg.pad_size)

    persistence = np.clip(4.0 * g8_up / (g1 + EPS), 0.0, 1.0)
    edge_strength = g1 / (g1.max() + EPS)
    a_map = np.clip(persistence * np.sqrt(edge_strength + EPS), 0.0, 1.0)

    return {
        "Y": y_pad,
        "a_map": a_map.astype(np.float64),
        "g1": g1,
        "g8_up": g8_up,
        "cfg": cfg,
    }


def _patch_grid(arr: np.ndarray, patch: int) -> Tuple[np.ndarray, int, int]:
    """Return (N, patch, patch) and grid gh,gw."""
    h, w = arr.shape
    gh, gw = h // patch, w // patch
    trunc = arr[: gh * patch, : gw * patch]
    patches = trunc.reshape(gh, patch, gw, patch).transpose(0, 2, 1, 3).reshape(-1, patch, patch)
    return patches, gh, gw


def build_granular_balls(
    statistics: Dict,
    patch_size: int,
    n_balls: int,
    seed: int = 42,
) -> Dict:
    """Build Fixed-K granular balls on one scale. Returns patch-level fields."""
    y = statistics["Y"]
    a_map = statistics["a_map"]
    patches_y, gh, gw = _patch_grid(y, patch_size)
    patches_a, _, _ = _patch_grid(a_map, patch_size)
    n = patches_y.shape[0]
    mean_y = patches_y.reshape(n, -1).mean(axis=1)
    std_y = patches_y.reshape(n, -1).std(axis=1)
    a_p = patches_a.reshape(n, -1).mean(axis=1)

    ys = np.arange(gh)
    xs = np.arange(gw)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    nx = (xx.reshape(-1) / max(gw - 1, 1)).astype(np.float64)
    ny = (yy.reshape(-1) / max(gh - 1, 1)).astype(np.float64)

    desc = np.stack([mean_y, std_y, a_p, 0.5 * nx, 0.5 * ny], axis=1)
    # standardize
    mu = desc.mean(axis=0, keepdims=True)
    sigma = desc.std(axis=0, keepdims=True) + EPS
    desc_z = (desc - mu) / sigma

    k = int(min(max(n_balls, 1), n))
    labels = _fixed_kmeans(desc_z, k, seed=seed)

    L_b = np.zeros(k, dtype=np.float64)
    U_b = np.zeros(k, dtype=np.float64)
    B_b = np.zeros(k, dtype=np.float64)
    mean_b = np.zeros(k, dtype=np.float64)
    for b in range(k):
        memb = a_p[labels == b]
        if memb.size == 0:
            continue
        L_b[b] = np.percentile(memb, 10)
        U_b[b] = np.percentile(memb, 90)
        B_b[b] = U_b[b] - L_b[b]
        mean_b[b] = memb.mean()

    return {
        "labels": labels,
        "a_p": a_p,
        "gh": gh,
        "gw": gw,
        "patch_size": patch_size,
        "L_b": L_b,
        "U_b": U_b,
        "B_b": B_b,
        "mean_b": mean_b,
        "n_balls": k,
    }


def _mask_from_balls(balls: Dict, mode: str, cfg: GranularMaskConfig) -> np.ndarray:
    labels = balls["labels"]
    a_p = balls["a_p"]
    L_b, U_b, B_b, mean_b = balls["L_b"], balls["U_b"], balls["B_b"], balls["mean_b"]
    n = a_p.shape[0]
    m = np.zeros(n, dtype=np.float64)

    if mode == "binary":
        med = float(np.median(mean_b)) if mean_b.size else 0.0
        for i in range(n):
            b = labels[i]
            m[i] = 1.0 if mean_b[b] >= med else 0.0
    elif mode == "fuzzy":
        for i in range(n):
            b = labels[i]
            m[i] = np.clip(0.5 * a_p[i] + 0.5 * mean_b[b], 0.0, 1.0)
    elif mode == "fuzzy_rough":
        for i in range(n):
            b = labels[i]
            m[i] = np.clip(a_p[i] * L_b[b] / (U_b[b] + EPS) - cfg.boundary_penalty * B_b[b], 0.0, 1.0)
    else:
        raise ValueError(f"unknown mode: {mode}")

    gh, gw, ps = balls["gh"], balls["gw"], balls["patch_size"]
    grid = m.reshape(gh, gw)
    # expand patches to pad resolution
    full = np.repeat(np.repeat(grid, ps, axis=0), ps, axis=1)
    return full


def derive_masks(
    statistics: Dict,
    mode: str,
    cfg: Optional[GranularMaskConfig] = None,
) -> Dict[str, np.ndarray]:
    """Derive combined 286×286 mask for one mode (and optionally all)."""
    cfg = cfg or statistics.get("cfg") or GranularMaskConfig()
    if mode not in ("binary", "fuzzy", "fuzzy_rough", "all"):
        raise ValueError(mode)

    modes = ["binary", "fuzzy", "fuzzy_rough"] if mode == "all" else [mode]
    fine = build_granular_balls(statistics, cfg.fine_patch_size, cfg.fine_balls, seed=cfg.seed)
    coarse = build_granular_balls(statistics, cfg.coarse_patch_size, cfg.coarse_balls, seed=cfg.seed)

    out: Dict[str, np.ndarray] = {}
    for m in modes:
        mf = _mask_from_balls(fine, m, cfg)
        mc = _mask_from_balls(coarse, m, cfg)
        # align shapes (both should be 288)
        h = min(mf.shape[0], mc.shape[0], cfg.pad_size)
        w = min(mf.shape[1], mc.shape[1], cfg.pad_size)
        comb = np.clip(cfg.fine_weight * mf[:h, :w] + cfg.coarse_weight * mc[:h, :w], 0.0, 1.0)
        if h < cfg.pad_size or w < cfg.pad_size:
            pad = np.zeros((cfg.pad_size, cfg.pad_size), dtype=np.float64)
            pad[:h, :w] = comb
            comb = pad
        m286 = _resize_bilinear(comb, cfg.target_size, cfg.target_size)
        m286 = np.clip(m286, 0.0, 1.0).astype(np.float32)
        out[m] = m286

    out["fine_balls"] = fine
    out["coarse_balls"] = coarse
    return out


def compute_mask_for_image(
    image: np.ndarray,
    mode: str = "all",
    cfg: Optional[GranularMaskConfig] = None,
) -> Dict[str, np.ndarray]:
    stats = compute_structural_statistics(image, cfg)
    return derive_masks(stats, mode=mode, cfg=cfg or GranularMaskConfig())
