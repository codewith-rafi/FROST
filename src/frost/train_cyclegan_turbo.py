import os
import gc
import copy
import re
import time
import lpips
import torch
import wandb
from datetime import datetime, timezone
from glob import glob
import numpy as np
from accelerate import Accelerator
from accelerate.utils import set_seed
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, CLIPTextModel
from diffusers.optimization import get_scheduler
from peft.utils import get_peft_model_state_dict
from cleanfid.fid import get_folder_features, build_feature_extractor, frechet_distance
import vision_aided_loss
from model import make_1step_sched
from cyclegan_turbo import CycleGAN_Turbo, VAE_encode, VAE_decode, initialize_unet, initialize_vae
from my_utils.dino_struct import DinoStructureLoss
from frost.cli_args import parse_args_unpaired_training
from frost.dataset import UnpairedDataset, build_transform
from frost.structure_loss import structure_preservation_loss


def _fmt_hms(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _load_generator_from_pkl(unet, vae_enc, vae_dec, ckpt_path: str) -> int:
    """Load CycleGAN-Turbo train pkl into existing UNet LoRA + VAE wrappers. Returns step from filename."""
    sd = torch.load(ckpt_path, map_location="cpu")
    for n, p in unet.named_parameters():
        name_sd = n.replace(".default_encoder.weight", ".weight")
        if "lora" in n and "default_encoder" in n:
            p.data.copy_(sd["sd_encoder"][name_sd])
    for n, p in unet.named_parameters():
        name_sd = n.replace(".default_decoder.weight", ".weight")
        if "lora" in n and "default_decoder" in n:
            p.data.copy_(sd["sd_decoder"][name_sd])
    for n, p in unet.named_parameters():
        name_sd = n.replace(".default_others.weight", ".weight")
        if "lora" in n and "default_others" in n:
            p.data.copy_(sd["sd_other"][name_sd])
    vae_enc.load_state_dict(sd["sd_vae_enc"])
    vae_dec.load_state_dict(sd["sd_vae_dec"])
    m = re.search(r"model_(\d+)\.pkl$", os.path.basename(ckpt_path))
    if not m:
        raise ValueError(f"cannot parse step from checkpoint name: {ckpt_path}")
    return int(m.group(1))


def main(args):
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, log_with=args.report_to)
    set_seed(args.seed)
    train_t0 = time.time()
    train_start_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    if accelerator.is_main_process:
        os.makedirs(os.path.join(args.output_dir, "checkpoints"), exist_ok=True)
        print(f"TRAIN_START {train_start_iso} output_dir={args.output_dir}")

    tokenizer = AutoTokenizer.from_pretrained("stabilityai/sd-turbo", subfolder="tokenizer", revision=args.revision, use_fast=False,)
    noise_scheduler_1step = make_1step_sched()
    text_encoder = CLIPTextModel.from_pretrained("stabilityai/sd-turbo", subfolder="text_encoder").cuda()

    unet, l_modules_unet_encoder, l_modules_unet_decoder, l_modules_unet_others = initialize_unet(args.lora_rank_unet, return_lora_module_names=True)
    vae_a2b, vae_lora_target_modules = initialize_vae(args.lora_rank_vae, return_lora_module_names=True)

    weight_dtype = torch.float32
    vae_a2b.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder.requires_grad_(False)

    if args.gan_disc_type == "vagan_clip":
        net_disc_a = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_a.cv_ensemble.requires_grad_(False)  # Freeze feature extractor
        net_disc_b = vision_aided_loss.Discriminator(cv_type='clip', loss_type=args.gan_loss_type, device="cuda")
        net_disc_b.cv_ensemble.requires_grad_(False)  # Freeze feature extractor

    crit_cycle, crit_idt = torch.nn.L1Loss(), torch.nn.L1Loss()

    if args.enable_xformers_memory_efficient_attention:
        unet.enable_xformers_memory_efficient_attention()

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    unet.conv_in.requires_grad_(True)
    vae_b2a = copy.deepcopy(vae_a2b)
    params_gen = CycleGAN_Turbo.get_traininable_params(unet, vae_a2b, vae_b2a)

    vae_enc = VAE_encode(vae_a2b, vae_b2a=vae_b2a)
    vae_dec = VAE_decode(vae_a2b, vae_b2a=vae_b2a)

    resume_step = 0
    if getattr(args, "resume_from", None):
        if not os.path.isfile(args.resume_from):
            raise FileNotFoundError(f"--resume_from not found: {args.resume_from}")
        resume_step = _load_generator_from_pkl(unet, vae_enc, vae_dec, args.resume_from)
        if resume_step >= args.max_train_steps:
            raise ValueError(
                f"resume_step={resume_step} >= max_train_steps={args.max_train_steps}; nothing to do"
            )
        if accelerator.is_main_process:
            print(
                f"RESUME_FROM {args.resume_from} resume_step={resume_step} "
                f"max_train_steps={args.max_train_steps} "
                f"(generator weights only; optimizer/disc cold-start)"
            )

    optimizer_gen = torch.optim.AdamW(params_gen, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

    params_disc = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
    optimizer_disc = torch.optim.AdamW(params_disc, lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay, eps=args.adam_epsilon,)

    dataset_train = UnpairedDataset(
        dataset_folder=args.dataset_folder,
        image_prep=args.train_img_prep,
        split="train",
        tokenizer=tokenizer,
        granular_mode=getattr(args, "granular_mode", "none"),
        granular_mask_dir=getattr(args, "granular_mask_dir", None),
    )
    dl_kwargs = dict(
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    # pin_memory + persistent_workers caused mid-run IPC death on BDD (~step 1960):
    # FileNotFoundError in resource_sharer → "Pin memory thread exited unexpectedly".
    # Keep workers=8 for load throughput; skip pin/persistent for long 2k-step stability.
    if args.dataloader_num_workers > 0:
        dl_kwargs.update(pin_memory=False, persistent_workers=False, prefetch_factor=2)
    train_dataloader = torch.utils.data.DataLoader(dataset_train, **dl_kwargs)
    T_val = build_transform(args.val_img_prep)
    fixed_caption_src = dataset_train.fixed_caption_src
    fixed_caption_tgt = dataset_train.fixed_caption_tgt
    l_images_src_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_src_test.extend(glob(os.path.join(args.dataset_folder, "test_A", ext)))
    l_images_tgt_test = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        l_images_tgt_test.extend(glob(os.path.join(args.dataset_folder, "test_B", ext)))
    l_images_src_test, l_images_tgt_test = sorted(l_images_src_test), sorted(l_images_tgt_test)

    # In-loop FID reference stats (skipped when validation_num_images==0; final eval is offline)
    feat_model = None
    a2b_ref_mu = a2b_ref_sigma = b2a_ref_mu = b2a_ref_sigma = None
    if accelerator.is_main_process and args.validation_num_images != 0:
        feat_model = build_feature_extractor("clean", "cuda", use_dataparallel=False)
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_a2b")
        os.makedirs(output_dir_ref, exist_ok=True)
        for _path in tqdm(l_images_tgt_test):
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert("RGB"))
                _img.save(outf)
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        a2b_ref_mu, a2b_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)
        output_dir_ref = os.path.join(args.output_dir, "fid_reference_b2a")
        os.makedirs(output_dir_ref, exist_ok=True)
        for _path in tqdm(l_images_src_test):
            outf = os.path.join(output_dir_ref, os.path.basename(_path)).replace(".jpg", ".png")
            if not os.path.exists(outf):
                _img = T_val(Image.open(_path).convert("RGB"))
                _img.save(outf)
        ref_features = get_folder_features(output_dir_ref, model=feat_model, num_workers=0, num=None,
                        shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", verbose=True,
                        custom_image_tranform=None)
        b2a_ref_mu, b2a_ref_sigma = np.mean(ref_features, axis=0), np.cov(ref_features, rowvar=False)

    lr_scheduler_gen = get_scheduler(args.lr_scheduler, optimizer=optimizer_gen,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)
    lr_scheduler_disc = get_scheduler(args.lr_scheduler, optimizer=optimizer_disc,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power)

    net_lpips = lpips.LPIPS(net='vgg')
    net_lpips.cuda()
    net_lpips.requires_grad_(False)

    fixed_a2b_tokens = tokenizer(fixed_caption_tgt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_a2b_emb_base = text_encoder(fixed_a2b_tokens.cuda().unsqueeze(0))[0].detach()
    fixed_b2a_tokens = tokenizer(fixed_caption_src, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids[0]
    fixed_b2a_emb_base = text_encoder(fixed_b2a_tokens.cuda().unsqueeze(0))[0].detach()
    del text_encoder, tokenizer  # free up some memory

    unet, vae_enc, vae_dec, net_disc_a, net_disc_b = accelerator.prepare(unet, vae_enc, vae_dec, net_disc_a, net_disc_b)
    net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc = accelerator.prepare(
        net_lpips, optimizer_gen, optimizer_disc, train_dataloader, lr_scheduler_gen, lr_scheduler_disc
    )
    if accelerator.is_main_process:
        accelerator.init_trackers(args.tracker_project_name, config=dict(vars(args)))

    first_epoch = 0
    global_step = resume_step
    progress_bar = tqdm(range(0, args.max_train_steps), initial=global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,)
    # turn off eff. attn for the disc
    for name, module in net_disc_a.named_modules():
        if "attn" in name:
            module.fused_attn = False
    for name, module in net_disc_b.named_modules():
        if "attn" in name:
            module.fused_attn = False

    for epoch in range(first_epoch, args.max_train_epochs):
        for step, batch in enumerate(train_dataloader):
            l_acc = [unet, net_disc_a, net_disc_b, vae_enc, vae_dec]
            with accelerator.accumulate(*l_acc):
                loss_granular_a2b = torch.zeros((), device=accelerator.device)
                loss_granular_b2a = torch.zeros((), device=accelerator.device)
                img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)

                bsz = img_a.shape[0]
                fixed_a2b_emb = fixed_a2b_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                fixed_b2a_emb = fixed_b2a_emb_base.repeat(bsz, 1, 1).to(dtype=weight_dtype)
                timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * bsz, device=img_a.device).long()

                """
                Cycle Objective
                """
                # A -> fake B -> rec A
                cyc_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                cyc_rec_a = CycleGAN_Turbo.forward_with_networks(cyc_fake_b, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                loss_cycle_a = crit_cycle(cyc_rec_a, img_a) * args.lambda_cycle
                loss_cycle_a += net_lpips(cyc_rec_a, img_a).mean() * args.lambda_cycle_lpips
                # B -> fake A -> rec B
                cyc_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                cyc_rec_b = CycleGAN_Turbo.forward_with_networks(cyc_fake_a, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_cycle_b = crit_cycle(cyc_rec_b, img_b) * args.lambda_cycle
                loss_cycle_b += net_lpips(cyc_rec_b, img_b).mean() * args.lambda_cycle_lpips
                accelerator.backward(loss_cycle_a + loss_cycle_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)

                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Generator Objective (GAN) for task a->b and b->a (fake inputs)
                """
                fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_gan_a = net_disc_a(fake_b, for_G=True).mean() * args.lambda_gan
                loss_gan_b = net_disc_b(fake_a, for_G=True).mean() * args.lambda_gan
                # Granular structure loss on existing fakes (no extra generator forward)
                loss_granular_a2b = torch.zeros((), device=img_a.device)
                loss_granular_b2a = torch.zeros((), device=img_a.device)
                if getattr(args, "lambda_granular", 0.0) and getattr(args, "granular_mode", "none") != "none":
                    mask_a = batch["granular_mask_src"].to(device=img_a.device, dtype=weight_dtype)
                    mask_b = batch["granular_mask_tgt"].to(device=img_b.device, dtype=weight_dtype)
                    loss_granular_a2b = structure_preservation_loss(img_a, fake_b, mask_a)
                    loss_granular_b2a = structure_preservation_loss(img_b, fake_a, mask_b)
                    loss_gan_a = loss_gan_a + args.lambda_granular * loss_granular_a2b
                    loss_gan_b = loss_gan_b + args.lambda_granular * loss_granular_b2a
                accelerator.backward(loss_gan_a + loss_gan_b, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()
                optimizer_disc.zero_grad()

                """
                Identity Objective
                """
                idt_a = CycleGAN_Turbo.forward_with_networks(img_b, "a2b", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_a2b_emb)
                loss_idt_a = crit_idt(idt_a, img_b) * args.lambda_idt
                loss_idt_a += net_lpips(idt_a, img_b).mean() * args.lambda_idt_lpips
                idt_b = CycleGAN_Turbo.forward_with_networks(img_a, "b2a", vae_enc, unet, vae_dec, noise_scheduler_1step, timesteps, fixed_b2a_emb)
                loss_idt_b = crit_idt(idt_b, img_a) * args.lambda_idt
                loss_idt_b += net_lpips(idt_b, img_a).mean() * args.lambda_idt_lpips
                loss_g_idt = loss_idt_a + loss_idt_b
                accelerator.backward(loss_g_idt, retain_graph=False)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(params_gen, args.max_grad_norm)
                optimizer_gen.step()
                lr_scheduler_gen.step()
                optimizer_gen.zero_grad()

                """
                Discriminator for task a->b and b->a (fake inputs)
                """
                loss_D_A_fake = net_disc_a(fake_b.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_B_fake = net_disc_b(fake_a.detach(), for_real=False).mean() * args.lambda_gan
                loss_D_fake = (loss_D_A_fake + loss_D_B_fake) * 0.5
                accelerator.backward(loss_D_fake, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

                """
                Discriminator for task a->b and b->a (real inputs)
                """
                loss_D_A_real = net_disc_a(img_b, for_real=True).mean() * args.lambda_gan
                loss_D_B_real = net_disc_b(img_a, for_real=True).mean() * args.lambda_gan
                loss_D_real = (loss_D_A_real + loss_D_B_real) * 0.5
                accelerator.backward(loss_D_real, retain_graph=False)
                if accelerator.sync_gradients:
                    params_to_clip = list(net_disc_a.parameters()) + list(net_disc_b.parameters())
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer_disc.step()
                lr_scheduler_disc.step()
                optimizer_disc.zero_grad()

            logs = {}
            logs["cycle_a"] = loss_cycle_a.detach().item()
            logs["cycle_b"] = loss_cycle_b.detach().item()
            logs["gan_a"] = loss_gan_a.detach().item()
            logs["gan_b"] = loss_gan_b.detach().item()
            logs["disc_a"] = loss_D_A_fake.detach().item() + loss_D_A_real.detach().item()
            logs["disc_b"] = loss_D_B_fake.detach().item() + loss_D_B_real.detach().item()
            logs["idt_a"] = loss_idt_a.detach().item()
            logs["idt_b"] = loss_idt_b.detach().item()
            logs["loss_granular_a2b"] = float(loss_granular_a2b.detach().item()) if torch.is_tensor(loss_granular_a2b) else 0.0
            logs["loss_granular_b2a"] = float(loss_granular_b2a.detach().item()) if torch.is_tensor(loss_granular_b2a) else 0.0
            if "granular_mask_src" in batch:
                logs["mask_mean_src"] = float(batch["granular_mask_src"].float().mean().item())
                logs["mask_mean_tgt"] = float(batch["granular_mask_tgt"].float().mean().item())
                logs["mask_mass_src"] = float(batch["granular_mask_src"].float().sum().item())
                logs["mask_mass_tgt"] = float(batch["granular_mask_tgt"].float().sum().item())

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    eval_unet = accelerator.unwrap_model(unet)
                    eval_vae_enc = accelerator.unwrap_model(vae_enc)
                    eval_vae_dec = accelerator.unwrap_model(vae_dec)
                    if args.viz_freq > 0 and global_step % args.viz_freq == 0:
                        for tracker in accelerator.trackers:
                            if tracker.name == "wandb":
                                viz_img_a = batch["pixel_values_src"].to(dtype=weight_dtype)
                                viz_img_b = batch["pixel_values_tgt"].to(dtype=weight_dtype)
                                log_dict = {
                                    "train/real_a": [wandb.Image(viz_img_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                    "train/real_b": [wandb.Image(viz_img_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)],
                                }
                                log_dict["train/rec_a"] = [wandb.Image(cyc_rec_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/rec_b"] = [wandb.Image(cyc_rec_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_b"] = [wandb.Image(fake_b[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                log_dict["train/fake_a"] = [wandb.Image(fake_a[idx].float().detach().cpu(), caption=f"idx={idx}") for idx in range(bsz)]
                                tracker.log(log_dict)
                                gc.collect()
                                torch.cuda.empty_cache()

                    # Checkpoint on step multiples and final step (§11.5: skip if steps==0)
                    if args.checkpointing_steps > 0 and (
                        (global_step % args.checkpointing_steps == 0) or (global_step == args.max_train_steps)
                    ):
                        outf = os.path.join(args.output_dir, "checkpoints", f"model_{global_step}.pkl")
                        sd = {}
                        sd["l_target_modules_encoder"] = l_modules_unet_encoder
                        sd["l_target_modules_decoder"] = l_modules_unet_decoder
                        sd["l_modules_others"] = l_modules_unet_others
                        sd["rank_unet"] = args.lora_rank_unet
                        sd["sd_encoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_encoder")
                        sd["sd_decoder"] = get_peft_model_state_dict(eval_unet, adapter_name="default_decoder")
                        sd["sd_other"] = get_peft_model_state_dict(eval_unet, adapter_name="default_others")
                        sd["rank_vae"] = args.lora_rank_vae
                        sd["vae_lora_target_modules"] = vae_lora_target_modules
                        sd["sd_vae_enc"] = eval_vae_enc.state_dict()
                        sd["sd_vae_dec"] = eval_vae_dec.state_dict()
                        torch.save(sd, outf)
                        gc.collect()
                        torch.cuda.empty_cache()

                    # In-loop FID/DINO: skip when validation_steps==0 or validation_num_images==0
                    run_val = (
                        args.validation_steps > 0
                        and args.validation_num_images != 0
                        and (
                            (global_step % args.validation_steps == 0)
                            or (global_step == args.max_train_steps)
                        )
                    )
                    if run_val:
                        _timesteps = torch.tensor([noise_scheduler_1step.config.num_train_timesteps - 1] * 1, device="cuda").long()
                        net_dino = DinoStructureLoss()
                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_a2b")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_a2b = []
                        for idx, input_img_path in enumerate(tqdm(l_images_src_test)):
                            if args.validation_num_images > 0 and idx >= args.validation_num_images:
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_a = transforms.ToTensor()(input_img)
                                img_a = transforms.Normalize([0.5], [0.5])(img_a).unsqueeze(0).cuda()
                                eval_fake_b = CycleGAN_Turbo.forward_with_networks(img_a, "a2b", eval_vae_enc, eval_unet,
                                    eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_a2b_emb[0:1])
                                eval_fake_b_pil = transforms.ToPILImage()(eval_fake_b[0] * 0.5 + 0.5)
                                eval_fake_b_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = net_dino.preprocess(eval_fake_b_pil).unsqueeze(0).cuda()
                                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                l_dino_scores_a2b.append(dino_ssim)
                        dino_score_a2b = np.mean(l_dino_scores_a2b)
                        gen_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                            shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                            custom_image_tranform=None)
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_a2b = frechet_distance(a2b_ref_mu, a2b_ref_sigma, ed_mu, ed_sigma)
                        _elapsed = time.time() - train_t0
                        print(
                            f"step={global_step}, elapsed_sec={_elapsed:.1f}, elapsed_hms={_fmt_hms(_elapsed)}, "
                            f"fid(a2b)={score_fid_a2b:.2f}, dino(a2b)={dino_score_a2b:.3f}"
                        )

                        fid_output_dir = os.path.join(args.output_dir, f"fid-{global_step}/samples_b2a")
                        os.makedirs(fid_output_dir, exist_ok=True)
                        l_dino_scores_b2a = []
                        for idx, input_img_path in enumerate(tqdm(l_images_tgt_test)):
                            if args.validation_num_images > 0 and idx >= args.validation_num_images:
                                break
                            outf = os.path.join(fid_output_dir, f"{idx}.png")
                            with torch.no_grad():
                                input_img = T_val(Image.open(input_img_path).convert("RGB"))
                                img_b = transforms.ToTensor()(input_img)
                                img_b = transforms.Normalize([0.5], [0.5])(img_b).unsqueeze(0).cuda()
                                eval_fake_a = CycleGAN_Turbo.forward_with_networks(img_b, "b2a", eval_vae_enc, eval_unet,
                                    eval_vae_dec, noise_scheduler_1step, _timesteps, fixed_b2a_emb[0:1])
                                eval_fake_a_pil = transforms.ToPILImage()(eval_fake_a[0] * 0.5 + 0.5)
                                eval_fake_a_pil.save(outf)
                                a = net_dino.preprocess(input_img).unsqueeze(0).cuda()
                                b = net_dino.preprocess(eval_fake_a_pil).unsqueeze(0).cuda()
                                dino_ssim = net_dino.calculate_global_ssim_loss(a, b).item()
                                l_dino_scores_b2a.append(dino_ssim)
                        dino_score_b2a = np.mean(l_dino_scores_b2a)
                        gen_features = get_folder_features(fid_output_dir, model=feat_model, num_workers=0, num=None,
                            shuffle=False, seed=0, batch_size=8, device=torch.device("cuda"),
                            mode="clean", custom_fn_resize=None, description="", verbose=True,
                            custom_image_tranform=None)
                        ed_mu, ed_sigma = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
                        score_fid_b2a = frechet_distance(b2a_ref_mu, b2a_ref_sigma, ed_mu, ed_sigma)
                        _elapsed = time.time() - train_t0
                        print(
                            f"step={global_step}, elapsed_sec={_elapsed:.1f}, elapsed_hms={_fmt_hms(_elapsed)}, "
                            f"fid(b2a)={score_fid_b2a}, dino(b2a)={dino_score_b2a:.3f}"
                        )
                        logs["val/fid_a2b"], logs["val/fid_b2a"] = score_fid_a2b, score_fid_b2a
                        logs["val/dino_struct_a2b"], logs["val/dino_struct_b2a"] = dino_score_a2b, dino_score_b2a
                        logs["elapsed_sec"] = _elapsed
                        del net_dino

            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            if global_step >= args.max_train_steps:
                break
        # Must also leave the epoch loop; otherwise each leftover epoch runs one more batch
        # (util gate: max_train_steps=100 → final_step=199 with max_train_epochs=100).
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        total = time.time() - train_t0
        end_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        print(
            f"TRAIN_END {end_iso} total_elapsed_sec={total:.1f} "
            f"total_elapsed_hms={_fmt_hms(total)} final_step={global_step}"
        )


if __name__ == "__main__":
    args = parse_args_unpaired_training()
    main(args)
