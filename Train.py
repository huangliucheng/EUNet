import os
import random
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import logging
import argparse

from torch.nn.utils import clip_grad_norm_
from torch import autocast
from torch.amp import GradScaler
from torchvision.utils import make_grid
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from model.model_factory import build_model
from utils.CODDataset import get_dataloader
from utils.evidence import evidence_to_prediction
from utils.Loss_Function import multi_loss_function as mlf


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


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)

        self.setup_logging()
        self.logger.info(f"Using device: {self.device}")
        if self.device.type == "cuda":
            self.logger.info(f"CUDA device name: {torch.cuda.get_device_name(self.device)}")
        self.build_data()
        self.build_model()

        self.global_step = 0
        self.mae_save_threshold = 0.0435
        self.best_maes = {'camo': 1.0}
        self.best_ema_maes = {'camo': 1.0}
        self.seed = int(cfg.seed)
        self.ema_decay = float(getattr(self.cfg.train, "ema_decay", 0.999))
        if not 0.0 < self.ema_decay < 1.0:
            raise ValueError("train.ema_decay must be between 0 and 1")
        self.ema_eval_start_epoch = self._ema_eval_start_epoch()
        self.scaler = GradScaler(device=self.device.type) # Mixed precision
        self.logger.info(
            f"EMA enabled from training start: decay={self.ema_decay}, "
            f"validation starts at epoch {self.ema_eval_start_epoch} after KL annealing reaches 1.0."
        )

    def setup_logging(self):
        os.makedirs(self.cfg.train.log_dir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(self.cfg.train.log_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M'
        )
        self.writer = SummaryWriter(os.path.join(self.cfg.train.tf_log_dir, 'tf-logs'))
        self.logger = logging.getLogger()
        self.logger.info(f"Configurations:\n{self.cfg}")

    def build_data(self):
        self.train_loader, self.train_num = get_dataloader(self.cfg, self.cfg.train.data_dir, csv_file=self.cfg.train.train_csv, mode='train')
        self.eval_loaders = {
            'camo': get_dataloader(self.cfg, self.cfg.eval.data_dir, csv_file=self.cfg.eval.camo_csv, mode='eval')[0],
        }

    def build_model(self):
        self.net = build_model(self.cfg).to(self.device)
        self.optimizer = self.build_optimizer()
        self.ema_net = copy.deepcopy(self.net).to(self.device)
        self.ema_net.eval()
        for param in self.ema_net.parameters():
            param.requires_grad_(False)

    def _kl_annealing_epochs(self):
        loss_cfg = getattr(self.cfg, "loss", None)
        kl_annealing_epochs = getattr(loss_cfg, "kl_annealing_epochs", None)
        if kl_annealing_epochs is None:
            return int(self.cfg.train.num_epochs) // 2

        kl_annealing_epochs = int(kl_annealing_epochs)
        if kl_annealing_epochs <= 0:
            raise ValueError("loss.kl_annealing_epochs must be a positive integer")
        return kl_annealing_epochs

    def _ema_eval_start_epoch(self):
        warmup_epochs = int(getattr(self.cfg.train, "warmup_epochs", 0) or 0)
        return warmup_epochs + self._kl_annealing_epochs() + 1

    def _ema_eval_ready(self, epoch):
        return (epoch + 1) >= self.ema_eval_start_epoch

    @torch.no_grad()
    def update_ema_model(self):
        model_state = self.net.state_dict()
        ema_state = self.ema_net.state_dict()
        for name, ema_value in ema_state.items():
            model_value = model_state[name].detach()
            if torch.is_floating_point(ema_value):
                ema_value.mul_(self.ema_decay).add_(model_value, alpha=1.0 - self.ema_decay)
            else:
                ema_value.copy_(model_value)

    def _mae_checkpoint_path(self, dataset_name):
        return os.path.join(
            self.cfg.train.save_dir,
            f"EUNet_{dataset_name}_{self.seed}.pth",
        )

    def _ema_checkpoint_path(self):
        return os.path.join(
            self.cfg.train.save_dir,
            f"EUNet_ema_{self.seed}.pth",
        )

    def _get_raw_backbone_module(self):
        encoder = self.net.encoder
        for attr_name in ("model", "swin", "pvt"):
            module = getattr(encoder, attr_name, None)
            if isinstance(module, torch.nn.Module):
                return module
        return encoder

    def _train_cfg_float(self, name, default):
        value = getattr(self.cfg.train, name, None)
        return float(default if value is None else value)

    def _skip_weight_decay(self, full_name, param, backbone_name=None, backbone_module=None):
        if param.ndim <= 1 or full_name.endswith(".bias"):
            return True

        no_decay_names = set()
        no_decay_keywords = set()
        if backbone_module is not None:
            if hasattr(backbone_module, "no_weight_decay"):
                no_decay_names = set(backbone_module.no_weight_decay())
            if hasattr(backbone_module, "no_weight_decay_keywords"):
                no_decay_keywords = set(backbone_module.no_weight_decay_keywords())

        if backbone_name in no_decay_names:
            return True
        if backbone_name and any(keyword in backbone_name for keyword in no_decay_keywords):
            return True

        default_keywords = (
            "absolute_pos_embed",
            "relative_position_bias_table",
            "pos_embed",
            "cls_token",
            "logit_scale",
        )
        return any(keyword in full_name for keyword in default_keywords)

    def build_optimizer(self):
        base_lr = self._train_cfg_float("base_lr", 1e-4)
        fallback_weight_decay = self._train_cfg_float("weight_decay", 1e-4)
        backbone_lr_scale = self._train_cfg_float("backbone_lr_scale", 0.2)
        backbone_weight_decay = self._train_cfg_float("backbone_weight_decay", fallback_weight_decay)
        custom_weight_decay = self._train_cfg_float("head_weight_decay", fallback_weight_decay)

        backbone_module = self._get_raw_backbone_module()
        backbone_param_ids = {id(param) for param in backbone_module.parameters()}
        backbone_param_names = {
            id(param): name for name, param in backbone_module.named_parameters()
        }

        grouped_params = {
            "backbone_decay": [],
            "backbone_no_decay": [],
            "custom_decay": [],
            "custom_no_decay": [],
        }
        grouped_numels = {name: 0 for name in grouped_params}

        for name, param in self.net.named_parameters():
            if not param.requires_grad:
                continue

            is_backbone = id(param) in backbone_param_ids
            backbone_name = backbone_param_names.get(id(param))
            no_decay = self._skip_weight_decay(
                name,
                param,
                backbone_name=backbone_name,
                backbone_module=backbone_module if is_backbone else None,
            )

            if is_backbone:
                group_name = "backbone_no_decay" if no_decay else "backbone_decay"
            else:
                group_name = "custom_no_decay" if no_decay else "custom_decay"

            grouped_params[group_name].append(param)
            grouped_numels[group_name] += param.numel()

        backbone_numel = grouped_numels["backbone_decay"] + grouped_numels["backbone_no_decay"]
        custom_numel = grouped_numels["custom_decay"] + grouped_numels["custom_no_decay"]
        if backbone_numel == 0 or custom_numel == 0:
            raise RuntimeError(
                "Failed to split optimizer parameters into backbone and custom groups."
            )

        param_groups = [
            {
                "name": "backbone_decay",
                "params": grouped_params["backbone_decay"],
                "lr": base_lr * backbone_lr_scale,
                "lr_scale": backbone_lr_scale,
                "weight_decay": backbone_weight_decay,
            },
            {
                "name": "backbone_no_decay",
                "params": grouped_params["backbone_no_decay"],
                "lr": base_lr * backbone_lr_scale,
                "lr_scale": backbone_lr_scale,
                "weight_decay": 0.0,
            },
            {
                "name": "custom_decay",
                "params": grouped_params["custom_decay"],
                "lr": base_lr,
                "lr_scale": 1.0,
                "weight_decay": custom_weight_decay,
            },
            {
                "name": "custom_no_decay",
                "params": grouped_params["custom_no_decay"],
                "lr": base_lr,
                "lr_scale": 1.0,
                "weight_decay": 0.0,
            },
        ]
        param_groups = [group for group in param_groups if group["params"]]

        for group in param_groups:
            self.logger.info(
                f"Optimizer group {group['name']}: "
                f"params={grouped_numels[group['name']]}, "
                f"lr={group['lr']:.2e}, "
                f"weight_decay={group['weight_decay']:.2e}"
            )

        return optim.AdamW(param_groups)

    def adjust_learning_rate(self, epoch):
        base_lr = self._train_cfg_float("base_lr", 1e-4)
        warmup_epochs = getattr(self.cfg.train, 'warmup_epochs', 0)
        power = 0.9

        if warmup_epochs > 0 and epoch < warmup_epochs:
            lr = base_lr * ((epoch + 1) / warmup_epochs)
        else:
            decay_ratio = (epoch - warmup_epochs) / (self.cfg.train.num_epochs - warmup_epochs)
            lr = base_lr * (1.0 - decay_ratio) ** power

        group_lrs = {}
        for param_group in self.optimizer.param_groups:
            lr_scale = param_group.get('lr_scale', 1.0)
            param_group['lr'] = lr * lr_scale
            group_lrs[param_group.get('name', 'default')] = param_group['lr']
        self.writer.add_scalar('Train/Learning_Rate', lr, self.global_step)
        self.writer.add_scalars('Train/Learning_Rate_Groups', group_lrs, self.global_step)
        return lr

    def train_one_epoch(self, epoch):
        self.net.train()
        running_loss = 0.0
        cur_lr = self.adjust_learning_rate(epoch)

        for i, data in enumerate(self.train_loader):
            self.global_step += 1
            
            images = data['image'].to(self.device, non_blocking=True)
            labels = data['label'].to(self.device, non_blocking=True)
            edges = data['edge'].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, dtype=torch.float16):
                preds_dict = self.net(images)
                
                warmup_epochs = getattr(self.cfg.train, 'warmup_epochs', 0)
                loss, loss_dict = mlf(
                    preds_dict,
                    labels,
                    edges,
                    epoch,
                    warmup_epochs,
                    self.cfg.train.num_epochs,
                    loss_cfg=getattr(self.cfg, "loss", None),
                )

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.net.parameters(), max_norm=self.cfg.train.clip_grad, norm_type=2)
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= previous_scale:
                self.update_ema_model()

            running_loss += loss.item()

            if self.global_step % self.cfg.train.print_freq == 0:
                self.log_train_info(epoch, i, running_loss, loss_dict, images, labels, edges, preds_dict)
                
        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def evaluate(self, epoch, dataset_name, use_ema=False):
        eval_net = self.ema_net if use_ema else self.net
        eval_net.eval()
        total_mae = 0.0
        dataloader = self.eval_loaders[dataset_name]
        
        for data in dataloader:
            images = data['image'].to(self.device, non_blocking=True)
            labels = data['label'].to(self.device, non_blocking=True)
            imsize = data['imsize']

            target_size = (imsize[1].item(), imsize[0].item())

            preds_dict = eval_net(images)
            evidence = preds_dict['final_evidence']

            pred = evidence_to_prediction(evidence, self.cfg)
            pred_for_mae = F.interpolate(pred, size=target_size, mode='bilinear')
            labels = F.interpolate(labels, size=target_size, mode='bilinear')

            total_mae += torch.mean(torch.abs(pred_for_mae - labels)).item()

        mae = total_mae / len(dataloader)
        eval_label = "EMA" if use_ema else "MODEL"
        writer_prefix = "Eval_EMA" if use_ema else "Eval"
        self.logger.info(
            f"[{dataset_name.upper()}][{eval_label}] Epoch: {epoch+1} | "
            f"Current MAE: {mae:.4f}"
        )
        self.writer.add_scalar(f'{writer_prefix}/{dataset_name}_MAE', mae, epoch+1)

        if use_ema:
            if mae < self.best_ema_maes[dataset_name]:
                self.best_ema_maes[dataset_name] = mae
                self.logger.info(f"[{dataset_name.upper()}][EMA] New Best MAE: {mae:.4f}!")
                if mae < self.mae_save_threshold:
                    save_path = self._ema_checkpoint_path()
                    torch.save(self.ema_net.state_dict(), save_path)
                    self.logger.info(
                        f"[{dataset_name.upper()}][EMA] Saved EMA checkpoint: {save_path}"
                    )
            return

        if mae < self.best_maes[dataset_name]:
            self.best_maes[dataset_name] = mae
            self.logger.info(f"[{dataset_name.upper()}][MODEL] New Best MAE: {mae:.4f}!")
            if mae < self.mae_save_threshold:
                save_path = self._mae_checkpoint_path(dataset_name)
                torch.save(self.net.state_dict(), save_path)
                self.logger.info(
                    f"[{dataset_name.upper()}][MODEL] Saved MAE checkpoint: {save_path}"
                )

    def log_train_info(self, epoch, batch_idx, running_loss, loss_dict, images, labels, edges, preds_dict):
        avg_loss = running_loss / (batch_idx + 1)
        self.logger.info(f"[Epoch {epoch+1}/{self.cfg.train.num_epochs}] "
                         f"Batch {batch_idx+1}/{len(self.train_loader)} | "
                         f"Avg Loss: {avg_loss:.4f}")
        
        self.writer.add_scalars('Train/Losses', loss_dict, self.global_step)

        vis_freq = getattr(self.cfg.train, 'vis_freq', self.cfg.train.print_freq * 5)
        if self.global_step % vis_freq == 0:
            self.writer.add_image('Vis/1_RGB', make_grid(images[0:1], normalize=True), self.global_step)
            self.writer.add_image('Vis/2_GT', make_grid(labels[0:1], normalize=True), self.global_step)

            final_pred = preds_dict['final_evidence']
            pred = evidence_to_prediction(final_pred, self.cfg)
            p = pred[0, 0, :, :].detach().cpu()
            self.writer.add_image('Vis/3_Pred', p, self.global_step, dataformats='HW')

            edge_gt = edges[0, 0, :, :].detach().cpu()
            self.writer.add_image('Vis/4_Edge_GT', edge_gt, self.global_step, dataformats='HW')

            edge_pred = preds_dict.get('edge_pred')
            if edge_pred is not None:
                edge_prob = torch.sigmoid(edge_pred[0, 0, :, :]).detach().cpu()
                self.writer.add_image('Vis/5_Edge_Pred', edge_prob, self.global_step, dataformats='HW')

            # Log prediction maps from each branch.
            branch_evidences = preds_dict.get('branch_evidence', [])
            for b_idx, b_ev in enumerate(branch_evidences):
                b_pred = evidence_to_prediction(b_ev, self.cfg)
                bp = b_pred[0, 0, :, :].detach().cpu()
                self.writer.add_image(f'Vis/6_Pred_Branch_{b_idx}', bp, self.global_step, dataformats='HW')

    def run(self):
        self.logger.info("=== Start Training ===")
        for epoch in range(self.cfg.train.num_epochs):
            epoch_loss = self.train_one_epoch(epoch)
            
            if (epoch + 1) >= self.cfg.train.eval_start_epoch: # Evaluation start epoch is configured in config.
                self.evaluate(epoch, 'camo')
                if self._ema_eval_ready(epoch):
                    self.evaluate(epoch, 'camo', use_ema=True)
                else:
                    self.logger.info(
                        f"[CAMO][EMA] Skip validation at epoch {epoch+1}; "
                        f"EMA validation starts at epoch {self.ema_eval_start_epoch}."
                    )
                
                if (epoch + 1) % self.cfg.train.save_freq == 0:
                    torch.save(self.net.state_dict(), 
                               os.path.join(self.cfg.train.save_dir, f"EUNet_epoch_{epoch+1}.pth"))
        
        self.logger.info("=== Training Finished ===")
        self.writer.close()

def build_config(args):
    cfg = OmegaConf.load("config/base.yaml")

    backbone_cfg_path = f"config/backbone_config/{args.backbone}_config.yaml"
    backbone_cfg = OmegaConf.load(backbone_cfg_path)
    cfg = OmegaConf.merge(cfg, backbone_cfg)
    cfg.device = args.device
    cfg.backbone_alias = args.backbone
    if args.seed is not None:
        cfg.seed = int(args.seed)
    cfg.seed = int(cfg.seed)
    cfg.train.save_dir = os.path.join(str(cfg.train.save_dir), str(cfg.seed))
    os.makedirs(cfg.train.save_dir, exist_ok=True)

    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="pvt",
                        choices=["swin", "swinv2", "res2net", "pvt", "convnext"])
    parser.add_argument("--device", type=str, default="cuda:0", help="Training device")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed; defaults to seed in config/base.yaml")
    args = parser.parse_args()

    cfg = build_config(args)
    set_seed(cfg.seed)

    trainer = Trainer(cfg)
    trainer.run()
    
    # os.system("/usr/bin/shutdown")
