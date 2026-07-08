# EUNet: Evidential Multi-Evidence Fusion for Uncertainty-Aware Camouflaged Object Detection

[English](README.md) | [中文](README_CN.md)

论文标题：**EUNet: Evidential Multi-Evidence Fusion for Uncertainty-Aware Camouflaged Object Detection**

本仓库是该论文的代码实现，包含 EUNet 的训练脚本、推理脚本、配置文件、模型实现和数据集 CSV 模板，用于不确定性感知的伪装目标检测。

## 1. 项目概述

EUNet 是一个面向伪装目标检测的证据多分支融合网络。本仓库支持：

- 使用不同 backbone 配置训练 EUNet。
- 显式加载模型参数进行推理。
- 在每个测试数据集对应目录下保存预测图和不确定性图。
- 对 CUCOD 数据集保留子类目录结构。

主要入口文件为：

```text
Train.py
Test.py
```

## 2. 环境配置

建议使用 Conda 创建独立虚拟环境：

```bash
conda create -n eunet python=3.12
conda activate eunet
```

请根据本机 CUDA 版本安装合适的 PyTorch。建议版本：

```text
torch>=2.11
torchvision
```

安装其余依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：

```text
numpy
pandas
Pillow
omegaconf
timm
tensorboard
opencv-python-headless
```

## 3. 数据准备

数据集路径在以下文件中配置：

```text
config/base.yaml
```

训练数据使用以下参数：

```text
train.data_dir
train.train_csv
```

训练 CSV 需要包含：

```text
img_path,gt_path,edge_path
```

测试数据使用以下参数：

```text
eval.data_dir
eval.camo_csv
eval.chameleon_csv
eval.cod10k_csv
eval.nc4k_csv
eval.cucod_csv
```

测试 CSV 需要包含：

```text
img_path,gt_path
```

CSV 中的路径会相对于对应的数据根目录进行解析。

## 4. 预训练权重和模型参数

backbone 预训练权重放置在：

```text
pretrained_ckpt/
```

当前 backbone 配置文件默认使用以下权重文件名：

```text
pretrained_ckpt/swinv2_base_patch4_window12_192_22k.pth
pretrained_ckpt/pvt_v2_b4.pth
pretrained_ckpt/res2net50_v1b_26w_4s-3cf99910.pth
pretrained_ckpt/convnextv2_base_22k_384_ema.pt
```

训练好的 EUNet 参数文件可以放在任意位置，并在运行 `Test.py` 时通过 `--checkpoint` 显式指定。

如果不指定 `--checkpoint`，`Test.py` 只会在 `config/base.yaml` 的 `train.save_dir` 下查找以下文件：

```text
EUNet_<backbone>.pth
EUNet_<backbone>_ema.pth
```

例如使用 `--backbone pvt` 时，默认候选文件为：

```text
snapshot/EUNet_pvt.pth
snapshot/EUNet_pvt_ema.pth
```

## 5. 训练

训练命令示例：

```bash
conda activate eunet
python Train.py --backbone pvt --device cuda:0 --seed 3407
```

支持的常用 backbone 配置名称：

```text
swin
pvt
res2net
convnext
```

训练参数默认保存到：

```text
snapshot/<seed>/
```

示例参数文件名：

```text
snapshot/3407/EUNet_camo_3407.pth
snapshot/3407/EUNet_ema_3407.pth
```

## 6. 测试

推荐显式指定参数文件进行推理：

```bash
conda activate eunet
python Test.py --backbone pvt --checkpoint snapshot/3407/EUNet_camo_3407.pth
```

推理结果默认保存到：

```text
results/EUNet/<backbone>/<dataset>/
```

每个数据集目录下包含：

```text
Pred/
Uncertainty/
```

`Pred/` 保存预测图，`Uncertainty/` 保存归一化后的不确定性图。

对于 CUCOD，脚本会保留子类目录结构，例如：

```text
results/EUNet/pvt/CUCOD/Pred/Appearance/
results/EUNet/pvt/CUCOD/Uncertainty/Appearance/
```

## 7. 配置文件

基础配置文件为：

```text
config/base.yaml
```

主要控制：

- `device`: 使用的设备，例如 `cuda:0`。
- `seed`: 随机种子。
- `train.data_dir`: 训练数据根目录。
- `train.train_csv`: 训练 CSV 路径。
- `train.save_dir`: 参数保存目录。
- `train.log_dir`: 训练日志目录。
- `train.tf_log_dir`: TensorBoard 日志目录。
- `eval.data_dir`: 测试数据根目录。
- `eval.*_csv`: 各测试集 CSV 路径。
- `loss.*`: 损失权重，包括边缘损失、主分支 EDL/KL 损失和辅助分支损失。
- `inference.output_mode`: 推理输出模式。

backbone 配置文件位于：

```text
config/backbone_config/
```

包括：

```text
config/backbone_config/swin_config.yaml
config/backbone_config/pvt_config.yaml
config/backbone_config/res2net_config.yaml
config/backbone_config/convnext_config.yaml
```

这些文件主要控制 backbone 类型、预训练权重路径、输入尺寸、batch size、学习率、weight decay 和 warmup epoch 等参数。

## 8. 下载链接

| 文件 | Google Drive | 百度网盘 |
| --- | --- | --- |
| Train | [Google Drive](https://drive.google.com/file/d/1G0FcQuvMLr1x_aSZHF2Ryb3ODDf6SCqZ/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/12H2-jtGvLCbtfcjV_6YWTA?pwd=hm9a)，提取码：`hm9a` |
| Test | [Google Drive](https://drive.google.com/file/d/1fpVrTsrwD5LPL-NWFJod64HyluHSQjM5/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1t8QueAIExSaa5V-268ju8w?pwd=m68m)，提取码：`m68m` |
| CUCOD | [Google Drive](https://drive.google.com/file/d/1JlCfohiP6iKERt5eOTeIjyBcgE-LgMkf/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1tGeEAikiiRSMOYZywWgBsg?pwd=b74z)，提取码：`b74z` |
| 预训练权重 | [Google Drive](https://drive.google.com/drive/folders/1EiA1rO2QJrmQFYIgWotAzlM5T7ktbhyX?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1S7IbRNuymw8EuLFPQIp8fg?pwd=mq14)，提取码：`mq14` |
| 参数文件 | [Google Drive](https://drive.google.com/file/d/15_7azCxDhN7mRkvqy2iD5WDLyRy3VNzm/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/16iAaymbWwte680ZOGbtBYg?pwd=such)，提取码：`such` |
| 预测结果文件 | [Google Drive](https://drive.google.com/file/d/1bz2LtG_MjC4blYxBJJDuXK5rLWSNnzAH/view?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1tfx1x7GR8ug7G5DlAzGADQ?pwd=9kfy)，提取码：`9kfy` |

下载的预测结果文件可以解压到：

```text
results/EUNet/
```