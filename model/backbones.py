import math
from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import torch
import torch.nn as nn

from .backbones_model.swin_transformer_v2 import SwinTransformerV2 as swin_bacbone
from .backbones_model.pvt_v2 import pvt_v2_b2, pvt_v2_b4
from .backbones_model.res2net_v1b import res2net50_v1b_26w_4s as res2net_backbone
from .backbones_model.convnextv2 import convnextv2_base as convnext_backbone

@dataclass(frozen=True)
class BackboneSpec:
    name: str
    encoder: nn.Module
    out_channels: Sequence[int]
    feature_transform: Callable


# 维度转换函数
def _identity_feature(feat):
    if feat.dim() != 4:
        raise ValueError(f'CNN backbone feature must be a 4D feature map, got {feat.dim()}D.')
    return feat

def _token_to_feature(feat):
    if feat.dim() == 4:
        return feat

    if feat.dim() != 3:
        raise ValueError('Backbone feature must be a 3D token tensor or a 4D feature map.')

    batch_size, seq_len, channels = feat.size()
    size = int(math.sqrt(seq_len))

    if size * size != seq_len:
        raise ValueError('Token feature length must be a square number to reshape into a feature map.')

    return feat.transpose(1, 2).contiguous().view(-1, channels, size, size)

class LayerNorm2d(nn.Module):
    def __init__(self, channels):
        super(LayerNorm2d, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f'LayerNorm2d expects a 4D feature map, got {x.dim()}D.')

        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()

class Res2NetAdapter(nn.Module):
    def __init__(self, original_model):
        super(Res2NetAdapter, self).__init__()
        self.model = original_model
        self.out_channels = [64, 256, 512, 1024, 2048]
        self.norms = nn.ModuleList([nn.GroupNorm(32, dim) for dim in self.out_channels])

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        f0 = x
        f1 = self.model.layer1(f0)
        f2 = self.model.layer2(f1)
        f3 = self.model.layer3(f2)
        f4 = self.model.layer4(f3)
        return [norm(feat) for norm, feat in zip(self.norms, [f0, f1, f2, f3, f4])]

class ConvNextAdapter(nn.Module):
    def __init__(self, original_model):
        super(ConvNextAdapter, self).__init__()
        self.model = original_model
        self.out_channels = [128, 256, 512, 1024, 1024]
        self.norms = nn.ModuleList([LayerNorm2d(dim) for dim in self.out_channels])

    def forward(self, x):
        x = self.model.downsample_layers[0](x)
        x = self.model.stages[0](x)
        f0 = x

        x = self.model.downsample_layers[1](x)
        x = self.model.stages[1](x)
        f1 = x

        x = self.model.downsample_layers[2](x)
        x = self.model.stages[2](x)
        f2 = x

        x = self.model.downsample_layers[3](x)
        stage4_blocks = list(self.model.stages[3].children())
        if not stage4_blocks:
            return [norm(feat) for norm, feat in zip(self.norms, [f0, f1, f2, x, x])]

        x = stage4_blocks[0](x)
        f3 = x
        for block in stage4_blocks[1:]:
            x = block(x)
        f4 = x

        return [norm(feat) for norm, feat in zip(self.norms, [f0, f1, f2, f3, f4])]

class SwinAdapter(nn.Module):
    def __init__(self, original_model):
        super(SwinAdapter, self).__init__()
        self.swin = original_model

        embed_dim = self.swin.embed_dim
        self.out_channels = [
            embed_dim, embed_dim * 2, embed_dim * 4, embed_dim * 8, embed_dim * 8
        ]
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in self.out_channels])

    def forward(self, x):
        x = self.swin.patch_embed(x)
        if self.swin.ape:
            x = x + self.swin.absolute_pos_embed
        x = self.swin.pos_drop(x)

        features = [self.norms[0](x)]
        for i, layer in enumerate(self.swin.layers):
            x = layer(x)
            features.append(self.norms[i + 1](x))

        return features

class PVTAdapter(nn.Module):
    def __init__(self, original_model):
        super(PVTAdapter, self).__init__()
        self.pvt = original_model
        self.out_channels = [64, 128, 320, 512, 512]
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for dim in self.out_channels])

    def forward(self, x):
        B = x.shape[0]

        x, H, W = self.pvt.patch_embed1(x)
        for blk in self.pvt.block1:
            x = blk(x, H, W)
        x = self.norms[0](x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        f0 = x

        x, H, W = self.pvt.patch_embed2(x)
        for blk in self.pvt.block2:
            x = blk(x, H, W)
        x = self.norms[1](x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        f1 = x

        x, H, W = self.pvt.patch_embed3(x)
        for blk in self.pvt.block3:
            x = blk(x, H, W)
        x = self.norms[2](x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        f2 = x

        x, H, W = self.pvt.patch_embed4(x)
        block4 = self.pvt.block4
        if len(block4) == 0:
            f3 = self._tokens_to_feature(self.norms[3](x), B, H, W)
            f4 = self._tokens_to_feature(self.norms[4](x), B, H, W)
            return [f0, f1, f2, f3, f4]

        x = block4[0](x, H, W)
        f3 = self._tokens_to_feature(self.norms[3](x), B, H, W)

        for blk in block4[1:]:
            x = blk(x, H, W)
        f4 = self._tokens_to_feature(self.norms[4](x), B, H, W)

        return [f0, f1, f2, f3, f4]

    @staticmethod
    def _tokens_to_feature(x, batch_size, height, width):
        return x.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()

def _build_res2net50(opt):
    model = res2net_backbone()
    if opt.pretrained:
        ckpt = torch.load(opt.weights_file, map_location=opt.device)
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
    adapter = Res2NetAdapter(model)
    return BackboneSpec('res2net50', adapter, adapter.out_channels, _identity_feature)

def _build_convnext_v2_base(opt):
    model = convnext_backbone()
    if opt.pretrained:
        ckpt = torch.load(opt.weights_file, map_location=opt.device)
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        model.load_state_dict(state_dict, strict=False)
    adapter = ConvNextAdapter(model)
    return BackboneSpec('convnext_v2_base', adapter, adapter.out_channels, _identity_feature)

def _build_swin_v2_base(opt):
    model = swin_bacbone(
        img_size=getattr(opt, 'img_size', 384),
        num_classes=0,
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=12,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )
    if opt.pretrained:
        ckpt = torch.load(opt.weights_file, map_location=opt.device)
        if 'model' in ckpt:
            checkpoint = ckpt['model']
        elif 'state_dict' in ckpt:
            checkpoint = ckpt['state_dict']
        else:
            checkpoint = ckpt
        model_state = model.state_dict()

        # 过滤掉因分辨率或窗口大小改变而导致形状不匹配的 Buffer (如 attn_mask 等)
        for k in list(checkpoint.keys()):
            if k in model_state and checkpoint[k].shape != model_state[k].shape:
                # 形状不匹配时将其从预训练字典中删除，保留模型自身初始化时生成的对应 Buffer
                del checkpoint[k]

        model.load_state_dict(checkpoint, strict=False)
    adapter = SwinAdapter(model)
    return BackboneSpec('swin_v2_base', adapter, adapter.out_channels, _token_to_feature)

def _build_pvt_backbone(opt, model_cls, backbone_name):
    model = model_cls(pretrained=None)
    if opt.pretrained:
        ckpt = torch.load(opt.weights_file, map_location=opt.device)
        if 'model' in ckpt:
            state_dict = ckpt['model']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt

        model_state = model.state_dict()
        for k in list(state_dict.keys()):
            if k in model_state and state_dict[k].shape != model_state[k].shape:
                # 如果遇到其他由于分辨率或结构调整引起的 shape 不匹配，自动跳过
                del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
    adapter = PVTAdapter(model)
    return BackboneSpec(backbone_name, adapter, adapter.out_channels, _identity_feature)


def _build_pvt_v2_b2(opt):
    return _build_pvt_backbone(opt, pvt_v2_b2, 'pvt_v2_b2')


def _build_pvt_v2_b4(opt):
    return _build_pvt_backbone(opt, pvt_v2_b4, 'pvt_v2_b4')


BACKBONE_BUILDERS: Dict[str, Callable] = {
    'res2net50': _build_res2net50,
    'convnext_v2': _build_convnext_v2_base,
    'swin_v2': _build_swin_v2_base,
    'pvt_v2': _build_pvt_v2_b2,
    'pvt_v2_b2': _build_pvt_v2_b2,
    'pvt_v2_b4': _build_pvt_v2_b4,
}

def build_backbone(opt) -> BackboneSpec:
    backbone_name = getattr(opt, 'backbone', 'res2net50').lower()

    if backbone_name not in BACKBONE_BUILDERS:
        supported = ', '.join(sorted(BACKBONE_BUILDERS.keys()))
        raise ValueError(f'Unsupported backbone: {backbone_name}. Supported: {supported}')

    print(f"==> Successfully built backbone: [{backbone_name.upper()}]")
    return BACKBONE_BUILDERS[backbone_name](opt)
