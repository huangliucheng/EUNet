# EUNet: Evidential Multi-Evidence Fusion for Uncertainty-Aware Camouflaged Object Detection

[English](README.md) | [中文](README_CN.md)

Title: **EUNet: Evidential Multi-Evidence Fusion for Uncertainty-Aware Camouflaged Object Detection**

This repository provides the code implementation of the EUNet paper. It includes the training script, inference script, configuration files, model implementation, and dataset CSV templates used for uncertainty-aware camouflaged object detection.

## 1. Overview

EUNet is an evidential multi-evidence fusion network for camouflaged object detection. The implementation in this repository supports:

- Training EUNet with different backbone configurations.
- Running inference with explicit model checkpoint loading.
- Saving prediction maps and uncertainty maps for each test dataset.
- Preserving the subclass folder structure for CUCOD during inference.

The main entry points are:

```text
Train.py
Test.py
```

## 2. Requirements

We recommend using a Conda environment:

```bash
conda create -n eunet python=3.12
conda activate eunet
```

Install PyTorch according to your CUDA version. Recommended:

```text
torch>=2.11
torchvision
```

Install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Main packages include:

```text
numpy
pandas
Pillow
omegaconf
timm
tensorboard
opencv-python-headless
```

## 3. Data Preparation

Set dataset paths in:

```text
config/base.yaml
```

Use the following parameters for training data:

```text
train.data_dir
train.train_csv
```

The training CSV should contain:

```text
img_path,gt_path,edge_path
```

Use the following parameters for testing data:

```text
eval.data_dir
eval.camo_csv
eval.chameleon_csv
eval.cod10k_csv
eval.nc4k_csv
eval.cucod_csv
```

Each testing CSV should contain:

```text
img_path,gt_path
```

Paths in the CSV files are resolved relative to the corresponding data root.

## 4. Pretrained Weights and Checkpoints

Place backbone pretrained weights under:

```text
pretrained_ckpt/
```

The current backbone configuration files expect:

```text
pretrained_ckpt/swinv2_base_patch4_window12_192_22k.pth
pretrained_ckpt/pvt_v2_b4.pth
pretrained_ckpt/res2net50_v1b_26w_4s-3cf99910.pth
pretrained_ckpt/convnextv2_base_22k_384_ema.pt
```

Trained EUNet checkpoints can be placed anywhere and passed explicitly to `Test.py` with `--checkpoint`.

If `--checkpoint` is omitted, `Test.py` only searches the following checkpoint names under `train.save_dir` in `config/base.yaml`:

```text
EUNet_<backbone>.pth
EUNet_<backbone>_ema.pth
```

For example, with `--backbone pvt`, the default candidates are:

```text
snapshot/EUNet_pvt.pth
snapshot/EUNet_pvt_ema.pth
```

## 5. Training

Run training with:

```bash
conda activate eunet
python Train.py --backbone pvt --device cuda:0 --seed 3407
```

Supported backbone config names:

```text
swin
pvt
res2net
convnext
```

Training checkpoints are saved under:

```text
snapshot/<seed>/
```

Example checkpoint names:

```text
snapshot/3407/EUNet_camo_3407.pth
snapshot/3407/EUNet_ema_3407.pth
```

## 6. Testing

Run inference with an explicit checkpoint:

```bash
conda activate eunet
python Test.py --backbone pvt --checkpoint snapshot/3407/EUNet_camo_3407.pth
```

Inference results are saved to:

```text
results/EUNet/<backbone>/<dataset>/
```

Each dataset folder contains:

```text
Pred/
Uncertainty/
```

`Pred/` stores prediction maps. `Uncertainty/` stores normalized uncertainty maps.

For CUCOD, the subclass folder structure is preserved:

```text
results/EUNet/pvt/CUCOD/Pred/Appearance/
results/EUNet/pvt/CUCOD/Uncertainty/Appearance/
```

## 7. Configuration

The base configuration file is:

```text
config/base.yaml
```

It mainly controls:

- `device`: device name, such as `cuda:0`.
- `seed`: random seed.
- `train.data_dir`: training data root.
- `train.train_csv`: training CSV path.
- `train.save_dir`: checkpoint save directory.
- `train.log_dir`: training log directory.
- `train.tf_log_dir`: TensorBoard log directory.
- `eval.data_dir`: testing data root.
- `eval.*_csv`: testing CSV paths.
- `loss.*`: loss weights, including edge loss, main EDL/KL loss, and auxiliary branch loss.
- `inference.output_mode`: inference output mode.

Backbone-specific configuration files are stored under:

```text
config/backbone_config/
```

Available files include:

```text
config/backbone_config/swin_config.yaml
config/backbone_config/pvt_config.yaml
config/backbone_config/res2net_config.yaml
config/backbone_config/convnext_config.yaml
```

These files mainly define the backbone type, pretrained weight path, image size, batch size, learning rate, weight decay, and warmup epochs.

## 8. Download Links

| File | Google Drive | Baidu Netdisk |
| --- | --- | --- |
| Train | [Google Drive](https://drive.google.com/file/d/1G0FcQuvMLr1x_aSZHF2Ryb3ODDf6SCqZ/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/12H2-jtGvLCbtfcjV_6YWTA?pwd=hm9a), code: `hm9a` |
| Test | [Google Drive](https://drive.google.com/file/d/1fpVrTsrwD5LPL-NWFJod64HyluHSQjM5/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1t8QueAIExSaa5V-268ju8w?pwd=m68m), code: `m68m` |
| CUCOD | [Google Drive](https://drive.google.com/file/d/1JlCfohiP6iKERt5eOTeIjyBcgE-LgMkf/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1tGeEAikiiRSMOYZywWgBsg?pwd=b74z), code: `b74z` |
| Pretrained weights | [Google Drive](https://drive.google.com/drive/folders/1EiA1rO2QJrmQFYIgWotAzlM5T7ktbhyX?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1S7IbRNuymw8EuLFPQIp8fg?pwd=mq14), code: `mq14` |
| Model parameters | [Google Drive](https://drive.google.com/file/d/15_7azCxDhN7mRkvqy2iD5WDLyRy3VNzm/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/16iAaymbWwte680ZOGbtBYg?pwd=such), code: `such` |
| Prediction results | [Google Drive](https://drive.google.com/file/d/1bz2LtG_MjC4blYxBJJDuXK5rLWSNnzAH/view?usp=sharing) | [Baidu Netdisk](https://pan.baidu.com/s/1tfx1x7GR8ug7G5DlAzGADQ?pwd=9kfy), code: `9kfy` |

Downloaded prediction results can be extracted to:

```text
results/EUNet/
```
