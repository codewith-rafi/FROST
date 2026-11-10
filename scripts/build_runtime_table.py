#!/usr/bin/env python3
"""§20.11 GPU / runtime table built from run logs (no re-execution).

Sources:
  output/**/train.log   -> TRAIN_END total_elapsed_sec, final_step
  output/**/eval.log    -> EVAL_START / EVAL_END, per-image tqdm s/it

Two documented traps are handled explicitly:
  1. BDD `fuzzy` elapsed covers only the 500-step resume leg (its model_2000 came from
     model_1500 + 500 steps), so s/step divides by 500, not 2000.
  2. BDD `binary` measures 6.87 s/step against 20.2-22.3 elsewhere at an identical step
     count. Reported as measured; the cause was never instrumented, so it is not a
     speedup claim.
Also: the "100-step" util gate actually reached final_step=199 (pre-fix epoch-break bug).
"""
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

# run -> (dataset, method, executed steps for the elapsed window)
RUNS = [
    ("fresh_20261024/horse2zebra/baseline", "horse", "baseline", 2000),
    ("fresh_20261024/horse2zebra/binary", "horse", "binary", 2000),
    ("fresh_20261024/horse2zebra/fuzzy", "horse", "fuzzy", 2000),
    ("fresh_20261024/horse2zebra/fuzzy_rough", "horse", "fuzzy_rough", 2000),
    ("fresh_20261027/bdd100k_day_night/baseline", "bdd", "baseline", 2000),
    ("fresh_20261027/bdd100k_day_night/binary", "bdd", "binary", 2000),
    ("fresh_20261027/bdd100k_day_night/fuzzy", "bdd", "fuzzy", 500),  # resumed leg only
    ("fresh_20261027/bdd100k_day_night/fuzzy_rough", "bdd", "fuzzy_rough", 2000),
]
BENCH = ("fresh_20261024/horse2zebra/util_gate_100", 199)


def parse_train(p: Path) -> dict:
    t = p.read_text(errors="replace")
    m = re.search(r"TRAIN_END (\S+) total_elapsed_sec=([\d.]+) total_elapsed_hms=(\S+) final_step=(\d+)", t)
    if not m:
        return {}
    return {
        "end": m.group(1),
        "elapsed": float(m.group(2)),
        "hms": m.group(3),
        "final_step": int(m.group(4)),
    }


def parse_eval(p: Path) -> dict:
    t = p.read_text(errors="replace").replace("\r", "\n")
    out = {}
    s = re.search(r"EVAL_START (\S+)", t)
    e = re.search(r"EVAL_END (\S+)", t)
    if s:
        out["start"] = s.group(1)
    if e:
        out["end"] = e.group(1)
    if s and e:
        try:
            a = datetime.fromisoformat(s.group(1))
            b = datetime.fromisoformat(e.group(1))
            out["wall_min"] = (b - a).total_seconds() / 60.0
        except ValueError:
            pass
    for direction in ("a2b", "b2a"):
        rates = re.findall(direction + r":\s+100%\|[^\r\n]*?([\d.]+)s/it", t)
        if rates:
            out[f"{direction}_s_per_img"] = float(rates[-1])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "report" / "tables" / "runtime_protocol.md").resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    md = ["# GPU, protocol and measured runtime (§20.11)", ""]
    md.append("Built from `TRAIN_END` / `EVAL_START` / `EVAL_END` lines and tqdm rates in the run "
              "logs. Nothing here was re-executed.")
    md.append("")
    md.append("## Training (measured wall clock)")
    md.append("")
    md.append("| Dataset | Method | Steps (executed) | Wall | s/step | final_step | note |")
    md.append("|---|---|---:|---:|---:|---:|---|")
    train_rows = []
    for rel, ds, method, steps in RUNS:
        info = parse_train(root / "output" / rel / "train.log")
        if not info:
            md.append(f"| {ds} | {method} | {steps} | - | - | - | log missing |")
            continue
        s_per = info["elapsed"] / steps
        note = ""
        if steps != info["final_step"]:
            note = "elapsed covers only the resumed leg (model_1500 + this leg)"
        if method == "binary" and ds == "bdd":
            note = "3.1x faster than every other run at identical steps; cause not instrumented, not a speedup claim"
        md.append(
            f"| {ds} | {method} | {steps} | {info['hms']} | {s_per:.2f} | {info['final_step']} | {note} |"
        )
        train_rows.append((ds, method, steps, info["elapsed"], s_per))

    md.append("")
    md.append("## Benchmark (100-step gate)")
    md.append("")
    bi = parse_train(root / "output" / BENCH[0] / "train.log")
    if bi:
        md.append(f"- `util_gate_100`: wall {bi['hms']} over **{bi['final_step']}** steps = "
                  f"{bi['elapsed'] / bi['final_step']:.2f} s/step.")
        md.append(f"- Caveat: the run was configured for 100 steps but reached `final_step={bi['final_step']}` "
                  "because of the pre-fix epoch-break bug, so it is a 199-step measurement, not 100. "
                  "Its s/step is consistent with the 2000-step runs.")
    md.append("")

    md.append("## Evaluation (generation vs FID phases)")
    md.append("")
    md.append("| Dataset | Method | Wall | a2b gen | b2a gen | FID phase (wall - gen) | note |")
    md.append("|---|---|---:|---:|---:|---:|---|")
    for rel, ds, method, _ in RUNS:
        info = parse_eval(root / "output" / rel / "eval.log")
        if not info.get("wall_min"):
            md.append(f"| {ds} | {method} | - | - | - | - | eval log missing |")
            continue
        a = info.get("a2b_s_per_img")
        b = info.get("b2a_s_per_img")
        gen_min = (120 * a + 140 * b) / 60.0 if (a and b) else None
        contaminated = ds == "bdd" and method == "binary"
        if contaminated:
            fid_cell = "n/a"
            note = ("wall (599 min) contaminated by the Clean-FID Inception download stall (~3 h) "
                    "plus kill and relaunch, so no per-phase split is possible; generation rates "
                    "above come from the completed tqdm bars and are still valid")
        else:
            fid_cell = f"{info['wall_min'] - gen_min:.0f} min" if gen_min else "-"
            note = ""
        md.append(
            f"| {ds} | {method} | {info['wall_min']:.0f} min | "
            f"{f'{a:.2f} s/img' if a else '-'} | {f'{b:.2f} s/img' if b else '-'} | "
            f"{fid_cell} | {note} |"
        )
    md.append("")
    md.append("Generation rates are descriptive: they vary run to run with GPU contention "
              "(a concurrent job shared the host) and were not a controlled variable.")
    md.append("")

    md.append("## Protocol held constant across all eight runs")
    md.append("")
    md.append("| Item | Value |")
    md.append("|---|---|")
    md.append("| Backbone | `stabilityai/sd-turbo`, one diffusion step |")
    md.append("| GPU (training) | GPU 1, `CUDA_VISIBLE_DEVICES=1` (from `scripts/launch_train.sh`) |")
    md.append("| GPU (final eval) | GPU 0 (GPU 1 was occupied by another job during the evaluation window) |")
    md.append("| Precision | FP32 weights + TF32 matmul and cuDNN |")
    md.append("| Attention | native PyTorch SDPA; xformers never enabled |")
    md.append("| Batch / accum | 1 / 1 |")
    md.append("| LoRA rank | UNet 128, VAE 4 |")
    md.append("| Learning rate | 1e-5, `constant` schedule |")
    md.append("| Steps | 2000 (`max_train_epochs=1`) |")
    md.append("| Checkpointing | every 500 |")
    md.append("| Validation in-loop | disabled (`validation_num_images=0`) |")
    md.append("| Seed | 42 |")
    md.append("| DataLoader workers | 8 (horse), 0 (BDD) |")
    md.append("| Hardware | 2x RTX 5090 32 GB, driver 595.84, CUDA 13.2, torch 2.10.0+cu128, python 3.11.16 |")
    md.append("")

    md.append("## What is NOT recorded (do not infer it)")
    md.append("")
    md.append("- Exact `CUDA_VISIBLE_DEVICES` per individual run: taken from the launcher, not from per-run artifacts.")
    md.append("- Peak VRAM, steady-state GPU utilization and power draw were never instrumented.")
    md.append("- `metrics.json` records dataset/method/seed/checkpoint only; it has no precision or "
              "attention fields, so those are documented from the run configs, not from the metric file.")
    md.append("- Training wall clock includes model load, mask IO and teardown, so it is an upper "
              "bound on pure optimizer-step time.")
    md.append("")

    out.write_text("\n".join(md))
    print(f"wrote {out}")
    print(f"training rows: {len(train_rows)}")
    for ds, m, steps, elapsed, sp in train_rows:
        print(f"  {ds:6s} {m:12s} {elapsed:9.1f}s / {steps:4d} = {sp:6.2f} s/step")
    if bi:
        print(f"  benchmark   {bi['elapsed']:9.1f}s / {bi['final_step']:4d} = {bi['elapsed'] / bi['final_step']:6.2f} s/step")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
