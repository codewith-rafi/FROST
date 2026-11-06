#!/usr/bin/env python3
"""Tiny YAML-ish loader for frost course configs (no PyYAML required for launchers)."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def load_frost_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text()
    cfg: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].rstrip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        k, v = k.strip(), v.strip()
        if not k:
            continue
        if v in ("null", "Null", "~", ""):
            cfg[k] = None
        elif v.lower() in ("true", "false"):
            cfg[k] = v.lower() == "true"
        else:
            try:
                cfg[k] = ast.literal_eval(v)
            except Exception:
                cfg[k] = v
    return cfg


if __name__ == "__main__":
    import json
    import sys

    print(json.dumps(load_frost_config(sys.argv[1]), indent=2, default=str))
