#!/usr/bin/env bash
# Precache torchvision VGG16 for lpips.LPIPS(net='vgg'). Uses conda rafi only.
set -euo pipefail

PYTHON="${PYTHON:-/home/user0/.conda/envs/rafi/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: expected rafi python at $PYTHON" >&2
  exit 1
fi
case "$PYTHON" in
  */envs/rafi/bin/python) ;;
  *)
    echo "ERROR: refuse non-rafi python: $PYTHON" >&2
    exit 1
    ;;
esac

"$PYTHON" - <<'PY'
import os
from pathlib import Path
import torch
from torch.hub import load_state_dict_from_url

url = "https://download.pytorch.org/models/vgg16-397923af.pth"
ckpt_dir = Path(torch.hub.get_dir()) / "checkpoints"
ckpt_dir.mkdir(parents=True, exist_ok=True)
dest = ckpt_dir / "vgg16-397923af.pth"
print(f"PYTHON={os.environ.get('CONDA_DEFAULT_ENV','?')} exe={Path(__import__('sys').executable)}")
print(f"target={dest}")

# Force download into hub cache (hash-checked by torchvision URL name).
state = load_state_dict_from_url(url, progress=True, check_hash=True, model_dir=str(ckpt_dir))
print(f"keys={len(state)} size_bytes={dest.stat().st_size if dest.exists() else 'MISSING'}")
if not dest.exists() or dest.stat().st_size < 400_000_000:
    # Some torch versions store under model_dir with same name; search
    found = list(ckpt_dir.glob("vgg16*.pth"))
    print("found", found)
    if not found or found[0].stat().st_size < 400_000_000:
        raise SystemExit("VGG16 cache too small or missing")

import lpips
print("smoke LPIPS(net='vgg') ...")
net = lpips.LPIPS(net="vgg")
del net
print("LPIPS_SMOKE_OK")
PY
