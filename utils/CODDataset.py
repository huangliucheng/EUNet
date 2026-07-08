import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode, v2


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class CODDataset(Dataset):
    def __init__(self, opt, data_dir, csv_file, is_train=True):
        super(CODDataset, self).__init__()
        self.data_dir = data_dir
        self.data_info = pd.read_csv(csv_file)
        self.is_train = is_train
        self.edge_kernel_size = self._validate_edge_kernel(
            getattr(opt.train, "edge_kernel_size", 3)
        )
        self.edge_smooth_kernel_size = self._validate_optional_edge_kernel(
            getattr(opt.train, "edge_smooth_kernel_size", 5),
            "train.edge_smooth_kernel_size",
        )
        self.edge_smooth_sigma = float(getattr(opt.train, "edge_smooth_sigma", 1.0))
        self.edge_smooth_kernel = (
            self._build_gaussian_kernel(
                self.edge_smooth_kernel_size,
                self.edge_smooth_sigma,
            )
            if self.edge_smooth_kernel_size > 1
            else None
        )

        train_size = [opt.img_size, opt.img_size]
        normalize = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        self.train_spatial_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomApply([
                v2.RandomRotation(
                    degrees=15,
                    interpolation=InterpolationMode.BILINEAR,
                )
            ], p=0.2),
            v2.Resize(
                train_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ])
        self.train_image_transform = v2.Compose([
            v2.ToImage(),
            v2.ColorJitter(
                brightness=(0.5, 1.5),
                contrast=(0.5, 1.5),
                saturation=(0.0, 2.0),
            ),
            v2.RandomChoice([
                v2.RandomAdjustSharpness(sharpness_factor=i / 10.0, p=1.0)
                for i in range(31)
            ]),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
            v2.ToPureTensor(),
        ])
        self.eval_image_transform = v2.Compose([
            v2.ToImage(),
            v2.Resize(
                train_size,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ),
            v2.ToDtype(torch.float32, scale=True),
            normalize,
            v2.ToPureTensor(),
        ])

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        row = self.data_info.iloc[idx]
        image_path = self._join_path(row["img_path"])
        label_path = self._join_path(row["gt_path"])

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to load image: {image_path}") from e

        label = Image.open(label_path).convert("L")
        imsize = image.size  # (W, H)

        if self.is_train:
            image, label, edge = self._transform_train_sample(image, label)
        else:
            image = self.eval_image_transform(image)
            label = self._mask_to_tensor(tv_tensors.Mask(label))

        sample = {
            "image": image,
            "label": label,
            "imsize": imsize,
            "name": os.path.basename(row["img_path"]),
            # 保留 CSV 中的相对路径，供 CUCOD 等分层数据集按子类保存结果。
            "relative_path": str(row["img_path"]),
        }
        if self.is_train:
            sample["edge"] = edge
        return sample

    def _transform_train_sample(self, image, label):
        label = tv_tensors.Mask(label)

        label = self._resize_mask_like_image(label, image)
        image, label = self.train_spatial_transform(image, label)

        image = self.train_image_transform(image)
        label = self._mask_to_tensor(label)
        edge = self._generate_edge_from_label(label)
        return image, label, edge

    def _resize_mask_like_image(self, mask, image):
        image_size = image.size[::-1] if isinstance(image, Image.Image) else image.shape[-2:]
        if mask.shape[-2:] == tuple(image_size):
            return mask
        return v2.Resize(
            list(image_size),
            interpolation=InterpolationMode.NEAREST,
            antialias=False,
        )(mask)

    @staticmethod
    def _mask_to_tensor(mask):
        mask = mask.as_subclass(torch.Tensor)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return mask.to(torch.float32).div(255.0)

    @staticmethod
    def _validate_edge_kernel(kernel_size):
        kernel_size = int(kernel_size)
        if kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("train.edge_kernel_size must be an odd integer >= 3")
        return kernel_size

    @staticmethod
    def _validate_optional_edge_kernel(kernel_size, name):
        kernel_size = int(kernel_size)
        if kernel_size <= 1:
            return 0
        if kernel_size % 2 == 0:
            raise ValueError(f"{name} must be 0/1 to disable or an odd integer >= 3")
        if kernel_size < 3:
            raise ValueError(f"{name} must be 0/1 to disable or an odd integer >= 3")
        return kernel_size

    @staticmethod
    def _build_gaussian_kernel(kernel_size, sigma):
        if sigma <= 0:
            raise ValueError("train.edge_smooth_sigma must be > 0")

        radius = kernel_size // 2
        coords = torch.arange(-radius, radius + 1, dtype=torch.float32)
        yy = coords.view(-1, 1)
        xx = coords.view(1, -1)
        kernel = torch.exp(-(xx.pow(2) + yy.pow(2)) / (2 * sigma * sigma))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def _generate_edge_from_label(self, label):
        mask = (label > 0.5).to(dtype=label.dtype)
        pad = self.edge_kernel_size // 2

        padded = F.pad(
            mask.unsqueeze(0),
            (pad, pad, pad, pad),
            mode="constant",
            value=0,
        )
        dilated = F.max_pool2d(padded, kernel_size=self.edge_kernel_size, stride=1)
        eroded = -F.max_pool2d(-padded, kernel_size=self.edge_kernel_size, stride=1)
        edge = (dilated - eroded).clamp_(0.0, 1.0)
        edge = self._smooth_edge(edge)
        return edge.squeeze(0)

    def _smooth_edge(self, edge):
        if self.edge_smooth_kernel is None:
            return edge

        kernel = self.edge_smooth_kernel.to(dtype=edge.dtype, device=edge.device)
        pad = self.edge_smooth_kernel_size // 2
        edge = F.pad(edge, (pad, pad, pad, pad), mode="replicate")
        edge = F.conv2d(edge, kernel)

        max_value = edge.amax(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        edge = edge / max_value
        return edge.clamp_(0.0, 1.0)

    def _join_path(self, rel_path):
        return os.path.join(self.data_dir, rel_path.lstrip("/"))

    def _edge_path(self, row):
        if "edge_path" not in row or pd.isna(row["edge_path"]):
            return None
        return self._join_path(row["edge_path"])


def get_dataloader(opt, data_dir, csv_file, mode="train", persistent_workers=True):
    is_train = mode == "train"
    dataset = CODDataset(opt, data_dir, csv_file, is_train)
    num_workers = opt.num_workers
    generator = torch.Generator()
    generator.manual_seed(getattr(opt, "seed", 3407))

    loader_kwargs = {
        "batch_size": opt.train.batch_size if is_train else 1,
        "shuffle": is_train,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": is_train,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = getattr(opt.train, "prefetch_factor", 4)

    dataloader = DataLoader(dataset, **loader_kwargs)

    return dataloader, len(dataset)
