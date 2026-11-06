"""CLI granular flags parse (frost.cli_args)."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from frost.cli_args import parse_args_unpaired_training  # noqa: E402


def test_course_cli_defaults():
    argv = [
        "prog",
        "--dataset_folder", "data/horse2zebra",
        "--output_dir", "/tmp/out",
        "--tracker_project_name", "t",
        "--train_img_prep", "resize_286_randomcrop_256x256_hflip",
        "--val_img_prep", "no_resize",
    ]
    with mock.patch.object(sys, "argv", argv):
        args = parse_args_unpaired_training()
    assert args.granular_mode == "none"
    assert args.lambda_granular == 0.10
    assert args.dataloader_num_workers == 8
    assert args.viz_freq == 0
    assert args.validation_num_images == 0
    assert args.granular_fine_balls == 16
    assert args.granular_seed == 42


def test_granular_cli_override():
    argv = [
        "prog",
        "--dataset_folder", "data/horse2zebra",
        "--output_dir", "/tmp/out",
        "--tracker_project_name", "t",
        "--train_img_prep", "resize_286_randomcrop_256x256_hflip",
        "--val_img_prep", "no_resize",
        "--granular_mode", "fuzzy_rough",
        "--lambda_granular", "0.1",
        "--granular_mask_dir", "data/horse2zebra/masks",
        "--dataloader_num_workers", "8",
        "--viz_freq", "0",
        "--validation_num_images", "0",
    ]
    with mock.patch.object(sys, "argv", argv):
        args = parse_args_unpaired_training()
    assert args.granular_mode == "fuzzy_rough"
    assert args.granular_mask_dir.endswith("masks")
    assert args.dataloader_num_workers == 8
