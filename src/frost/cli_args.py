"""CLI for unpaired CycleGAN-Turbo (frost course)."""
from __future__ import annotations

import argparse


def parse_args_unpaired_training(input_args=None):
    parser = argparse.ArgumentParser(description="Frost CycleGAN-Turbo unpaired training.")

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_idt", default=1, type=float)
    parser.add_argument("--lambda_cycle", default=1, type=float)
    parser.add_argument("--lambda_cycle_lpips", default=10.0, type=float)
    parser.add_argument("--lambda_idt_lpips", default=1.0, type=float)

    parser.add_argument(
        "--granular_mode",
        type=str,
        default="none",
        choices=["none", "binary", "fuzzy", "fuzzy_rough"],
    )
    parser.add_argument("--lambda_granular", type=float, default=0.10)
    parser.add_argument("--granular_mask_dir", type=str, default=None)
    parser.add_argument("--granular_fine_patch_size", type=int, default=8)
    parser.add_argument("--granular_coarse_patch_size", type=int, default=16)
    parser.add_argument("--granular_fine_balls", type=int, default=16)
    parser.add_argument("--granular_coarse_balls", type=int, default=4)
    parser.add_argument("--granular_max_fine_balls", type=int, default=64)
    parser.add_argument("--granular_max_coarse_balls", type=int, default=16)
    parser.add_argument("--granular_binary_threshold", type=float, default=0.50)
    parser.add_argument("--granular_lower_percentile", type=float, default=10)
    parser.add_argument("--granular_upper_percentile", type=float, default=90)
    parser.add_argument("--granular_boundary_penalty", type=float, default=0.25)
    parser.add_argument("--granular_purity_threshold", type=float, default=0.80)
    parser.add_argument("--granular_fine_weight", type=float, default=0.60)
    parser.add_argument("--granular_coarse_weight", type=float, default=0.40)
    parser.add_argument("--granular_seed", type=int, default=42)

    parser.add_argument("--dataset_folder", required=True, type=str)
    parser.add_argument("--train_img_prep", required=True)
    parser.add_argument("--val_img_prep", required=True)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--max_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)

    parser.add_argument("--pretrained_model_name_or_path", default="stabilityai/sd-turbo")
    parser.add_argument("--revision", default=None, type=str)
    parser.add_argument("--variant", default=None, type=str)
    parser.add_argument("--lora_rank_unet", default=128, type=int)
    parser.add_argument("--lora_rank_vae", default=4, type=int)

    parser.add_argument("--viz_freq", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--tracker_project_name", type=str, required=True)
    parser.add_argument("--validation_steps", type=int, default=500)
    parser.add_argument(
        "--validation_num_images",
        type=int,
        default=0,
        help="In-loop val image count. 0=skip in-loop FID/DINO; -1=all; >0=cap.",
    )
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Load generator LoRA/VAE weights from a model_N.pkl and continue from step N. "
        "Does not restore optimizer or discriminator state.",
    )

    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--max_grad_norm", default=10.0, type=float)
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help='["linear","cosine","cosine_with_restarts","polynomial","constant","constant_with_warmup"]',
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")

    if input_args is not None:
        return parser.parse_args(input_args)
    return parser.parse_args()
