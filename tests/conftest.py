# pytest root helpers — frost package + vendor imports
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FROST_SRC = ROOT / "src"
VENDOR_SRC = ROOT / "vendor" / "img2img-turbo" / "src"
for p in (FROST_SRC, VENDOR_SRC):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)
