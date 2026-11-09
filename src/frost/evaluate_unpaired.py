#!/usr/bin/env python3
"""Offline unpaired eval: FID + DINO-Struct (instruction §15)."""
from __future__ import annotations

import argparse
import json
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance

from cyclegan_turbo import CycleGAN_Turbo
from my_utils.dino_struct import DinoStructureLoss
from frost.dataset import build_transform


def _list_images(folder: str) -> list[str]:
    paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        paths.extend(glob(os.path.join(folder, ext)))
    return sorted(paths)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--dataset_folder", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--metrics_json", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="horse2zebra")
    ap.add_argument("--method_name", type=str, default="baseline")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--image_prep", type=str, default="no_resize")
    ap.add_argument("--prompt_a", type=str, default=None)
    ap.add_argument("--prompt_b", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(args.dataset_folder, "fixed_prompt_a.txt")) as f:
        prompt_a = args.prompt_a or f.read().strip()
    with open(os.path.join(args.dataset_folder, "fixed_prompt_b.txt")) as f:
        prompt_b = args.prompt_b or f.read().strip()

    model = CycleGAN_Turbo(pretrained_path=args.checkpoint)
    model.eval()
    model.to(device)

    T_val = build_transform(args.image_prep)
    src_test = _list_images(os.path.join(args.dataset_folder, "test_A"))
    tgt_test = _list_images(os.path.join(args.dataset_folder, "test_B"))

    out_root = Path(args.output_dir) / "eval_final"
    dir_a2b = out_root / "samples_a2b"
    dir_b2a = out_root / "samples_b2a"
    ref_a2b = out_root / "ref_a2b"
    ref_b2a = out_root / "ref_b2a"
    for d in (dir_a2b, dir_b2a, ref_a2b, ref_b2a):
        d.mkdir(parents=True, exist_ok=True)

    net_dino = DinoStructureLoss()
    dino_a2b, dino_b2a = [], []

    for idx, path in enumerate(tqdm(src_test, desc="a2b")):
        pil = Image.open(path).convert("RGB")
        inp = T_val(pil)
        x = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(inp)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x, direction="a2b", caption=prompt_b)
        out_pil = transforms.ToPILImage()(out[0].cpu() * 0.5 + 0.5)
        out_pil.save(dir_a2b / f"{idx}.png")
        a = net_dino.preprocess(inp).unsqueeze(0).to(device)
        b = net_dino.preprocess(out_pil).unsqueeze(0).to(device)
        dino_a2b.append(float(net_dino.calculate_global_ssim_loss(a, b).item()))

    for idx, path in enumerate(tgt_test):
        Image.open(path).convert("RGB").save(ref_a2b / f"{idx}.png")

    for idx, path in enumerate(tqdm(tgt_test, desc="b2a")):
        pil = Image.open(path).convert("RGB")
        inp = T_val(pil)
        x = transforms.Normalize([0.5], [0.5])(transforms.ToTensor()(inp)).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x, direction="b2a", caption=prompt_a)
        out_pil = transforms.ToPILImage()(out[0].cpu() * 0.5 + 0.5)
        out_pil.save(dir_b2a / f"{idx}.png")
        a = net_dino.preprocess(inp).unsqueeze(0).to(device)
        b = net_dino.preprocess(out_pil).unsqueeze(0).to(device)
        dino_b2a.append(float(net_dino.calculate_global_ssim_loss(a, b).item()))

    for idx, path in enumerate(src_test):
        Image.open(path).convert("RGB").save(ref_b2a / f"{idx}.png")

    feat_model = build_feature_extractor("clean", str(device), use_dataparallel=False)

    def fid_pair(gen_dir, ref_dir):
        gen = get_folder_features(
            str(gen_dir),
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=device,
            mode="clean",
            custom_fn_resize=None,
            description="",
            verbose=True,
            custom_image_tranform=None,
        )
        ref = get_folder_features(
            str(ref_dir),
            model=feat_model,
            num_workers=0,
            num=None,
            shuffle=False,
            seed=0,
            batch_size=8,
            device=device,
            mode="clean",
            custom_fn_resize=None,
            description="",
            verbose=True,
            custom_image_tranform=None,
        )
        return float(
            frechet_distance(
                np.mean(ref, 0),
                np.cov(ref, rowvar=False),
                np.mean(gen, 0),
                np.cov(gen, rowvar=False),
            )
        )

    fid_a2b = fid_pair(dir_a2b, ref_a2b)
    fid_b2a = fid_pair(dir_b2a, ref_b2a)

    metrics = {
        "dataset": args.dataset_name,
        "method": args.method_name,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "a2b": {
            "fid": fid_a2b,
            "dino_struct_mean": float(np.mean(dino_a2b)) if dino_a2b else None,
            "dino_struct_std": float(np.std(dino_a2b)) if dino_a2b else None,
            "dino_struct_per_image": dino_a2b,
        },
        "b2a": {
            "fid": fid_b2a,
            "dino_struct_mean": float(np.mean(dino_b2a)) if dino_b2a else None,
            "dino_struct_std": float(np.std(dino_b2a)) if dino_b2a else None,
            "dino_struct_per_image": dino_b2a,
        },
    }
    Path(args.metrics_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_json).write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"fid_a2b={fid_a2b:.4f} dino_a2b={metrics['a2b']['dino_struct_mean']}")
    print(f"fid_b2a={fid_b2a:.4f} dino_b2a={metrics['b2a']['dino_struct_mean']}")
    print(f"Wrote {args.metrics_json}")


if __name__ == "__main__":
    main()
