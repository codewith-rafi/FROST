"""Fast coarse-gradient structure loss (instruction §10 Mag@4×)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def rgb_to_luminance(x: torch.Tensor) -> torch.Tensor:
    """x: (B,3,H,W) in [-1,1] → Y (B,1,H,W) in [0,1]."""
    x01 = (x + 1.0) * 0.5
    r, g, b = x01[:, 0:1], x01[:, 1:2], x01[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _sobel_grad(y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """y: (B,1,H,W) → Yx, Yy same shape."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=y.dtype, device=y.device).view(1, 1, 3, 3)
    yx = F.conv2d(y, kx, padding=1)
    yy = F.conv2d(y, ky, padding=1)
    return yx, yy


def _mag_hat(y4: torch.Tensor) -> torch.Tensor:
    """Normalized Sobel magnitude on 4× downsampled luminance."""
    yx, yy = _sobel_grad(y4)
    mag = torch.sqrt(yx ** 2 + yy ** 2 + EPS)
    mean = mag.mean(dim=(-2, -1), keepdim=True)
    flat = mean < 1e-6
    mag_hat = mag / (mean + EPS)
    return torch.where(flat.expand_as(mag_hat), torch.zeros_like(mag_hat), mag_hat)


def structure_preservation_loss(
    img_in: torch.Tensor,
    img_out: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """L_struct = mean(M_4 * |Mag_hat(Y_G_4) - Mag_hat(Y_x_4)|). No / mean(M)."""
    y_x = rgb_to_luminance(img_in)
    y_g = rgb_to_luminance(img_out)
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)
    y_x4 = F.avg_pool2d(y_x, kernel_size=4, stride=4)
    y_g4 = F.avg_pool2d(y_g, kernel_size=4, stride=4)
    m4 = F.avg_pool2d(mask.float(), kernel_size=4, stride=4)
    mag_x = _mag_hat(y_x4)
    mag_g = _mag_hat(y_g4)
    return (m4 * (mag_g - mag_x).abs()).mean()
