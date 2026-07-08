import os
import random
import argparse
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from omegaconf import OmegaConf

from model.model_factory import build_model
from utils.CODDataset import get_dataloader
from utils.evidence import evidence_to_prediction, evidence_to_probability, evidence_to_uncertainty

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Tester:
    SUPPORTED_DATASETS = ("CAMO", "CHAMELEON", "COD10K", "NC4K", "CUCOD")
    NPY_DATASETS = {"COD10K", "NC4K", "CUCOD"}

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.result_name = args.run_name or Path(args.checkpoint).stem

        self.setup_logging()
        self.build_data()
        self.build_model()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M'
        )
        self.logger = logging.getLogger()
        self.logger.info(f"Configurations:\n{self.cfg}")

    def build_data(self):
        self.test_datasets = {
            'CAMO': self.cfg.eval.camo_csv,
            'CHAMELEON': self.cfg.eval.chameleon_csv,
            'COD10K': self.cfg.eval.cod10k_csv,
            'NC4K': self.cfg.eval.nc4k_csv,
            'CUCOD': self.cfg.eval.cucod_csv,
        }
        self.test_datasets = {
            name: self.test_datasets[name]
            for name in self.args.datasets
        }
        self.test_loaders = {}
        for name, csv_path in self.test_datasets.items():
            # Each test loader is consumed once; keep no workers alive afterward.
            loader, _ = get_dataloader(
                self.cfg,
                self.cfg.eval.data_dir,
                csv_file=csv_path,
                mode='eval',
                persistent_workers=False,
            )
            self.test_loaders[name] = loader

    def build_model(self):
        self.net = build_model(self.cfg).to(self.device)
        self.logger.info(f"Loading checkpoint from: {self.args.checkpoint}")
        checkpoint = torch.load(self.args.checkpoint, map_location=self.device)

        if 'state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['state_dict'])
        else:
            self.net.load_state_dict(checkpoint)
        self.net.eval()

    @staticmethod
    def _result_subdir(dataset_name, relative_path):
        """Preserve the CUCOD Image/<subclass>/ hierarchy; save other datasets directly."""
        if dataset_name != "CUCOD":
            return Path()

        parts = Path(str(relative_path)).parts
        image_indices = [index for index, part in enumerate(parts[:-1]) if part == "Image"]
        if not image_indices:
            raise ValueError(f"CUCOD image path is missing the Image/<subclass> hierarchy: {relative_path}")

        subdir_parts = parts[image_indices[-1] + 1 : -1]
        if not subdir_parts:
            raise ValueError(f"CUCOD image path is missing the subclass directory: {relative_path}")
        return Path(*subdir_parts)

    @staticmethod
    def _artifact_path(root, subdir, filename):
        output_dir = Path(root) / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / filename

    @torch.no_grad()
    def test_dataset(self, dataset_name, dataloader):
        self.logger.info(f"[{dataset_name}] Start Testing...")

        # Use one top-level directory per checkpoint to avoid overwriting weights from the same backbone.
        base_save_path = os.path.join(self.args.save_dir, self.result_name, dataset_name)
        save_path_pred = os.path.join(base_save_path, 'Pred')
        save_path_prob_raw = os.path.join(base_save_path, 'Pred_Projected', 'Raw')
        save_path_unc_raw = os.path.join(base_save_path, 'Uncertainty', 'Raw')
        save_full_outputs = self.args.backbone == 'swin'
        save_uncertainty_images = self.args.save_uncertainty_images or save_full_outputs
        save_npy_outputs = dataset_name in self.NPY_DATASETS

        os.makedirs(save_path_pred, exist_ok=True)
        if save_npy_outputs:
            os.makedirs(save_path_prob_raw, exist_ok=True)
            os.makedirs(save_path_unc_raw, exist_ok=True)
        if save_uncertainty_images:
            save_path_unc_norm = os.path.join(base_save_path, 'Uncertainty', 'Normalized')
            os.makedirs(save_path_unc_norm, exist_ok=True)
        if save_full_outputs:
            save_path_edge = os.path.join(base_save_path, 'Edge')
            os.makedirs(save_path_edge, exist_ok=True)

        for i, data in enumerate(dataloader):
            images = data['image'].to(self.device)
            imsize = data['imsize']
            names = data['name']
            relative_paths = data.get('relative_path', names)

            target_size = (imsize[1].item(), imsize[0].item())

            preds_dict = self.net(images)
            evidence = preds_dict['final_evidence']
            edge_pred = preds_dict.get('edge_pred') if save_full_outputs else None
            branch_evidences = preds_dict.get('branch_evidence', []) if save_full_outputs else []

            pred = evidence_to_prediction(evidence, self.cfg)
            projected_prob = evidence_to_probability(evidence)[:, 0:1, :, :] if save_npy_outputs else None
            uncertainty = evidence_to_uncertainty(evidence)
            pred = F.interpolate(pred, size=target_size, mode='bilinear')
            if projected_prob is not None:
                projected_prob = F.interpolate(projected_prob, size=target_size, mode='bilinear')
            unc_map = F.interpolate(uncertainty, size=target_size, mode='bilinear')

            if edge_pred is not None:
                edge_pred = F.interpolate(edge_pred, size=target_size, mode='bilinear')
                edge_pred = torch.sigmoid(edge_pred).cpu().numpy()

            pred = pred.cpu().numpy()
            if projected_prob is not None:
                projected_prob = projected_prob.cpu().numpy()
            unc_map = unc_map.cpu().numpy()

            branch_preds = []
            for b_ev in branch_evidences:
                b_pred_map = evidence_to_prediction(b_ev, self.cfg)
                b_pred = F.interpolate(b_pred_map, size=target_size, mode='bilinear')
                branch_preds.append(b_pred.cpu().numpy())

            for j, name in enumerate(names):
                base_name = os.path.splitext(name)[0]
                save_name = base_name + '.png'
                raw_name = base_name + '.npy'
                result_subdir = self._result_subdir(dataset_name, relative_paths[j])

                # 1. Save Prediction Map
                p = pred[j, 0]
                p = np.clip(p * 255, 0, 255).astype(np.uint8)
                pred_path = self._artifact_path(save_path_pred, result_subdir, save_name)
                cv2.imwrite(str(pred_path), p)

                u_raw = unc_map[j, 0].astype(np.float32)
                if save_npy_outputs:
                    # Keep raw numeric maps; float16 reduces disk usage while preserving np.load compatibility.
                    prob_save = projected_prob[j, 0].astype(np.float16)
                    unc_save = u_raw.astype(np.float16)
                    prob_path = self._artifact_path(save_path_prob_raw, result_subdir, raw_name)
                    unc_path = self._artifact_path(save_path_unc_raw, result_subdir, raw_name)
                    np.save(prob_path, prob_save)
                    np.save(unc_path, unc_save)

                if save_uncertainty_images:
                    u_min, u_max = np.min(u_raw), np.max(u_raw)
                    if u_max > u_min:
                        u_norm = (u_raw - u_min) / (u_max - u_min)
                    else:
                        u_norm = np.zeros_like(u_raw)
                    u_norm = np.clip(u_norm * 255, 0, 255).astype(np.uint8)
                    u_color = cv2.applyColorMap(u_norm, cv2.COLORMAP_JET)
                    unc_image_path = self._artifact_path(
                        save_path_unc_norm,
                        result_subdir,
                        save_name,
                    )
                    cv2.imwrite(str(unc_image_path), u_color)

                if not save_full_outputs:
                    continue

                # 3. Save Edge Prediction Map
                if edge_pred is not None:
                    e = edge_pred[j, 0]
                    e = np.clip(e * 255, 0, 255).astype(np.uint8)
                    edge_path = self._artifact_path(save_path_edge, result_subdir, save_name)
                    cv2.imwrite(str(edge_path), e)

                # 4. Save Branch Prediction Maps
                for b_idx, b_pred_map in enumerate(branch_preds):
                    branch_dir = os.path.join(base_save_path, f'Pred_branch_{b_idx}')
                    bp = b_pred_map[j, 0]
                    bp = np.clip(bp * 255, 0, 255).astype(np.uint8)
                    branch_path = self._artifact_path(branch_dir, result_subdir, save_name)
                    cv2.imwrite(str(branch_path), bp)

            if (i + 1) % 50 == 0:
                self.logger.info(f"[{dataset_name}] Processed {i + 1}/{len(dataloader)} images.")

        self.logger.info(f"[{dataset_name}] Finished Testing. Saved to {base_save_path}")

    def run(self):
        self.logger.info("=== Start Testing ===")
        for dataset_name, dataloader in self.test_loaders.items():
            self.test_dataset(dataset_name, dataloader)
        self.logger.info("=== All Testing Finished ===")

def build_config(backbone_name, output_mode=None):
    repo_root = Path(__file__).resolve().parent
    cfg = OmegaConf.load(repo_root / "config/base.yaml")
    backbone_cfg_path = repo_root / "config" / "backbone_config" / f"{backbone_name}_config.yaml"
    backbone_cfg = OmegaConf.load(backbone_cfg_path)
    cfg = OmegaConf.merge(cfg, backbone_cfg)
    if output_mode is not None:
        cfg.inference.output_mode = output_mode
    return cfg

if __name__ == '__main__':
    set_seed(3407)

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="pvt",
                        choices=["swin", "swinv2", "res2net", "pvt", "convnext"],
                        help="Backbone name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "EUNet_results" / "results"),
        help="Root directory for inference results; each checkpoint creates a separate subdirectory",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Result subdirectory name; defaults to the checkpoint filename without extension",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=Tester.SUPPORTED_DATASETS,
        default=list(Tester.SUPPORTED_DATASETS[:-1]),
        help="Datasets to run inference on; defaults to CAMO CHAMELEON COD10K NC4K",
    )
    parser.add_argument(
        "--save_uncertainty_images",
        action="store_true",
        help="Save normalized uncertainty heatmaps; non-Swin models such as ConvNeXt require this flag",
    )
    parser.add_argument(
        "--output_mode",
        type=str,
        default=None,
        choices=["foreground_belief", "projected_probability"],
        help="Override inference.output_mode for F0 output-form comparison",
    )
    args = parser.parse_args()

    cfg = build_config(args.backbone, args.output_mode)

    tester = Tester(cfg, args)
    tester.run()
