"""Unpaired dataset with aligned image/mask transforms (instruction §9)."""
from __future__ import annotations

import os
import random
from glob import glob

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F


def build_transform(image_prep: str):
    if image_prep == "resized_crop_512":
        return transforms.Compose(
            [
                transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.CenterCrop(512),
            ]
        )
    if image_prep == "resize_286_randomcrop_256x256_hflip":
        return transforms.Compose(
            [
                transforms.Resize((286, 286), interpolation=Image.LANCZOS),
                transforms.RandomCrop((256, 256)),
                transforms.RandomHorizontalFlip(),
            ]
        )
    if image_prep in ["resize_256", "resize_256x256"]:
        return transforms.Compose([transforms.Resize((256, 256), interpolation=Image.LANCZOS)])
    if image_prep in ["resize_512", "resize_512x512"]:
        return transforms.Compose([transforms.Resize((512, 512), interpolation=Image.LANCZOS)])
    if image_prep == "no_resize":
        return transforms.Lambda(lambda x: x)
    raise ValueError(f"unknown image_prep: {image_prep}")


def _load_mask_tensor(mask_path: str) -> torch.Tensor:
    if mask_path.endswith(".npz"):
        data = np.load(mask_path)
        arr = data["mask"] if "mask" in data else data[data.files[0]]
        t = torch.from_numpy(np.asarray(arr, dtype=np.float32))
    elif mask_path.endswith(".pt"):
        obj = torch.load(mask_path, map_location="cpu")
        t = obj["mask"] if isinstance(obj, dict) else obj
        t = t.float()
    elif mask_path.endswith(".npy"):
        t = torch.from_numpy(np.load(mask_path).astype(np.float32))
    else:
        raise FileNotFoundError(mask_path)
    if t.dim() == 2:
        t = t.unsqueeze(0)
    return t.clamp(0, 1)


def _aligned_train_pair(img_pil: Image.Image, mask_t: torch.Tensor | None):
    img = F.resize(img_pil, [286, 286], interpolation=transforms.InterpolationMode.LANCZOS)
    img_t = F.to_tensor(img)
    if mask_t is None:
        mask_t = torch.ones(1, 286, 286, dtype=torch.float32)
    else:
        if mask_t.shape[-2:] != (286, 286):
            mask_t = F.resize(mask_t, [286, 286], interpolation=transforms.InterpolationMode.BILINEAR)
        mask_t = mask_t.float().clamp(0, 1)

    i, j, h, w = transforms.RandomCrop.get_params(img_t, output_size=(256, 256))
    img_t = F.crop(img_t, i, j, h, w)
    mask_t = F.crop(mask_t, i, j, h, w)
    if random.random() < 0.5:
        img_t = F.hflip(img_t)
        mask_t = F.hflip(mask_t)
    img_t = F.normalize(img_t, mean=[0.5], std=[0.5])
    return img_t, mask_t


class UnpairedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_folder,
        split,
        image_prep,
        tokenizer,
        granular_mode="none",
        granular_mask_dir=None,
    ):
        super().__init__()
        if split == "train":
            self.source_folder = os.path.join(dataset_folder, "train_A")
            self.target_folder = os.path.join(dataset_folder, "train_B")
            self.src_split_name = "train_A"
            self.tgt_split_name = "train_B"
        elif split == "test":
            self.source_folder = os.path.join(dataset_folder, "test_A")
            self.target_folder = os.path.join(dataset_folder, "test_B")
            self.src_split_name = "test_A"
            self.tgt_split_name = "test_B"
        else:
            raise ValueError(split)

        self.tokenizer = tokenizer
        self.granular_mode = granular_mode or "none"
        self.granular_mask_dir = granular_mask_dir
        self.image_prep = image_prep
        self.use_aligned = image_prep == "resize_286_randomcrop_256x256_hflip"

        with open(os.path.join(dataset_folder, "fixed_prompt_a.txt"), "r") as f:
            self.fixed_caption_src = f.read().strip()
            self.input_ids_src = self.tokenizer(
                self.fixed_caption_src,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids

        with open(os.path.join(dataset_folder, "fixed_prompt_b.txt"), "r") as f:
            self.fixed_caption_tgt = f.read().strip()
            self.input_ids_tgt = self.tokenizer(
                self.fixed_caption_tgt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids

        self.l_imgs_src = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_src.extend(glob(os.path.join(self.source_folder, ext)))
        self.l_imgs_tgt = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.gif"]:
            self.l_imgs_tgt.extend(glob(os.path.join(self.target_folder, ext)))
        self.T = build_transform(image_prep)

    def __len__(self):
        return len(self.l_imgs_src) + len(self.l_imgs_tgt)

    def _mask_path_for(self, img_path: str, split_name: str) -> str | None:
        if self.granular_mode == "none" or not self.granular_mask_dir:
            return None
        stem = os.path.splitext(os.path.basename(img_path))[0]
        cand = os.path.join(
            self.granular_mask_dir, split_name, f"{stem}_mask_{self.granular_mode}.npz"
        )
        if os.path.isfile(cand):
            return cand
        raise FileNotFoundError(f"missing mask for {img_path}: expected {cand}")

    def __getitem__(self, index):
        if index < len(self.l_imgs_src):
            img_path_src = self.l_imgs_src[index]
        else:
            img_path_src = random.choice(self.l_imgs_src)
        img_path_tgt = random.choice(self.l_imgs_tgt)
        img_pil_src = Image.open(img_path_src).convert("RGB")
        img_pil_tgt = Image.open(img_path_tgt).convert("RGB")

        if self.use_aligned:
            m_src_path = self._mask_path_for(img_path_src, self.src_split_name)
            m_tgt_path = self._mask_path_for(img_path_tgt, self.tgt_split_name)
            m_src = _load_mask_tensor(m_src_path) if m_src_path else None
            m_tgt = _load_mask_tensor(m_tgt_path) if m_tgt_path else None
            img_t_src, mask_src = _aligned_train_pair(img_pil_src, m_src)
            img_t_tgt, mask_tgt = _aligned_train_pair(img_pil_tgt, m_tgt)
        else:
            img_t_src = F.normalize(F.to_tensor(self.T(img_pil_src)), mean=[0.5], std=[0.5])
            img_t_tgt = F.normalize(F.to_tensor(self.T(img_pil_tgt)), mean=[0.5], std=[0.5])
            mask_src = torch.ones(1, img_t_src.shape[-2], img_t_src.shape[-1])
            mask_tgt = torch.ones(1, img_t_tgt.shape[-2], img_t_tgt.shape[-1])

        return {
            "pixel_values_src": img_t_src,
            "pixel_values_tgt": img_t_tgt,
            "granular_mask_src": mask_src,
            "granular_mask_tgt": mask_tgt,
            "caption_src": self.fixed_caption_src,
            "caption_tgt": self.fixed_caption_tgt,
            "input_ids_src": self.input_ids_src,
            "input_ids_tgt": self.input_ids_tgt,
        }
