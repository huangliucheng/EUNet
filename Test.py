import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from model.model_factory import build_model
from utils.CODDataset import get_dataloader
from utils.evidence import evidence_to_prediction, evidence_to_uncertainty


SUPPORTED_BACKBONES = ("swin", "pvt", "res2net", "convnext")
DATASET_CONFIG_KEYS = {
    "CAMO": "camo_csv",
    "CHAMELEON": "chameleon_csv",
    "COD10K": "cod10k_csv",
    "NC4K": "nc4k_csv",
    "CUCOD": "cucod_csv",
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device_name):
    device_name = str(device_name)
    if device_name.isdigit():
        device_name = f"cuda:{device_name}"

    device = torch.device(device_name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device was requested ({device_name}), but CUDA is not available.")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError(
                f"Requested CUDA device index {index}, but only {torch.cuda.device_count()} device(s) are visible."
            )
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    return device


def repo_root():
    return Path(__file__).resolve().parent


def resolve_path(path_like):
    path = Path(str(path_like)).expanduser()
    if path.is_absolute():
        return path
    return repo_root() / path


def build_config(backbone_name):
    root = repo_root()
    cfg = OmegaConf.load(root / "config/base.yaml")
    backbone_cfg_path = root / "config" / "backbone_config" / f"{backbone_name}_config.yaml"
    if not backbone_cfg_path.exists():
        raise FileNotFoundError(f"Backbone config not found: {backbone_cfg_path}")
    return OmegaConf.merge(cfg, OmegaConf.load(backbone_cfg_path))


def configured_datasets(cfg):
    datasets = {}
    for dataset_name, config_key in DATASET_CONFIG_KEYS.items():
        csv_value = getattr(cfg.eval, config_key, None)
        if csv_value is None:
            continue
        csv_path = resolve_path(csv_value)
        if csv_path.exists():
            datasets[dataset_name] = str(csv_path)
    if not datasets:
        raise RuntimeError("No valid evaluation CSV files were found from cfg.eval.")
    return datasets


def checkpoint_candidates(cfg, backbone_name):
    save_dir = resolve_path(cfg.train.save_dir)
    return [
        save_dir / f"EUNet_{backbone_name}.pth",
        save_dir / f"EUNet_{backbone_name}_ema.pth",
    ]


def resolve_checkpoint(cfg, backbone_name, checkpoint_path=None):
    if checkpoint_path:
        checkpoint = resolve_path(checkpoint_path)
        if checkpoint.exists():
            return checkpoint
        raise FileNotFoundError(f"Specified checkpoint does not exist: {checkpoint}")

    candidates = checkpoint_candidates(cfg, backbone_name)
    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "No checkpoint was found. Specify --checkpoint or place a supported file under train.save_dir. "
        f"Checked candidates:\n{checked}"
    )


def result_subdir(dataset_name, relative_path):
    if dataset_name != "CUCOD":
        return Path()

    parts = Path(str(relative_path)).parts
    image_indices = [index for index, part in enumerate(parts[:-1]) if part == "Image"]
    if not image_indices:
        raise ValueError(f"CUCOD image path lacks Image/<subclass> hierarchy: {relative_path}")

    subdir_parts = parts[image_indices[-1] + 1 : -1]
    if not subdir_parts:
        raise ValueError(f"CUCOD image path lacks subclass hierarchy: {relative_path}")
    return Path(*subdir_parts)


def artifact_path(root, subdir, filename):
    output_dir = Path(root) / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename


def normalize_to_uint8(uncertainty):
    u_min = np.min(uncertainty)
    u_max = np.max(uncertainty)
    if u_max > u_min:
        normalized = (uncertainty - u_min) / (u_max - u_min)
    else:
        normalized = np.zeros_like(uncertainty)
    return np.clip(normalized * 255, 0, 255).astype(np.uint8)


class EUNetInferencer:
    def __init__(self, cfg, backbone_name, checkpoint_path=None):
        self.cfg = cfg
        self.backbone_name = backbone_name
        self.checkpoint_path = checkpoint_path
        self.device = resolve_device(cfg.device)
        self.output_root = repo_root() / "results" / "EUNet" / backbone_name

        self.setup_logging()
        self.datasets = configured_datasets(cfg)
        self.model = self.build_model()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M",
        )
        self.logger = logging.getLogger()
        self.logger.info(f"Configurations:\n{self.cfg}")

    def build_model(self):
        model = build_model(self.cfg).to(self.device)
        checkpoint_path = resolve_checkpoint(
            self.cfg,
            self.backbone_name,
            checkpoint_path=self.checkpoint_path,
        )
        self.logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def build_loader(self, csv_path):
        loader, _ = get_dataloader(
            self.cfg,
            self.cfg.eval.data_dir,
            csv_file=csv_path,
            mode="eval",
            persistent_workers=False,
        )
        return loader

    @torch.no_grad()
    def infer_dataset(self, dataset_name, csv_path):
        dataloader = self.build_loader(csv_path)
        dataset_root = self.output_root / dataset_name
        pred_root = dataset_root / "Pred"
        uncertainty_root = dataset_root / "Uncertainty"
        pred_root.mkdir(parents=True, exist_ok=True)
        uncertainty_root.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"[{dataset_name}] Start inference.")
        for index, data in enumerate(dataloader):
            images = data["image"].to(self.device)
            imsize = data["imsize"]
            names = data["name"]
            relative_paths = data.get("relative_path", names)
            target_size = (imsize[1].item(), imsize[0].item())

            outputs = self.model(images)
            evidence = outputs["final_evidence"]
            pred = evidence_to_prediction(evidence, self.cfg)
            uncertainty = evidence_to_uncertainty(evidence)

            pred = F.interpolate(pred, size=target_size, mode="bilinear", align_corners=False)
            uncertainty = F.interpolate(
                uncertainty,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )
            pred = pred.cpu().numpy()
            uncertainty = uncertainty.cpu().numpy()

            for batch_index, name in enumerate(names):
                base_name = os.path.splitext(name)[0]
                save_name = base_name + ".png"
                subdir = result_subdir(dataset_name, relative_paths[batch_index])

                pred_image = np.clip(pred[batch_index, 0] * 255, 0, 255).astype(np.uint8)
                uncertainty_image = normalize_to_uint8(uncertainty[batch_index, 0].astype(np.float32))

                Image.fromarray(pred_image).save(artifact_path(pred_root, subdir, save_name))
                Image.fromarray(uncertainty_image).save(artifact_path(uncertainty_root, subdir, save_name))

            if (index + 1) % 50 == 0:
                self.logger.info(f"[{dataset_name}] Processed {index + 1}/{len(dataloader)} images.")

        self.logger.info(f"[{dataset_name}] Finished. Saved to: {dataset_root}")

    def run(self):
        self.logger.info(f"Using device: {self.device}")
        self.logger.info(f"Saving results to: {self.output_root}")
        for dataset_name, csv_path in self.datasets.items():
            self.infer_dataset(dataset_name, csv_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="EUNet inference with config-controlled dataset, checkpoint, and output settings."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="swin",
        choices=SUPPORTED_BACKBONES,
        help="Backbone config name under config/backbone_config.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Explicit checkpoint path. If omitted, Test.py searches only "
            "EUNet_<backbone>.pth and EUNet_<backbone>_ema.pth under train.save_dir."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = build_config(args.backbone)
    set_seed(int(cfg.seed))
    inferencer = EUNetInferencer(cfg, args.backbone, checkpoint_path=args.checkpoint)
    inferencer.run()
