import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_backbone
from .BasicConv2d import BasicConv2d
from .BasicRFB import BasicRFB_a as RFB


class ResizeToMatch(nn.Module):
    def __init__(self, mode='bilinear', align_corners=False):
        super(ResizeToMatch, self).__init__()
        self.mode = mode
        # nearest 模式不支持 align_corners，做一下兼容处理
        self.align_corners = align_corners if mode in ['bilinear', 'bicubic'] else None

    def forward(self, src: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 只要高度或宽度不一致，就触发插值
        if src.shape[2:] != target.shape[2:]:
            return F.interpolate(
                src,
                size=target.shape[2:],
                mode=self.mode,
                align_corners=self.align_corners
            )
        return src

class FusionHead(nn.Module):
    def __init__(self, in_channels):
        super(FusionHead, self).__init__()
        final_conv = nn.Conv2d(in_channels//2, 2, kernel_size=3, stride=1, padding=1)

        nn.init.kaiming_normal_(final_conv.weight, mode='fan_out', nonlinearity='relu')
        if final_conv.bias is not None:
            nn.init.constant_(final_conv.bias, 0)

        self.head = nn.Sequential(
            BasicConv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1),
            final_conv,
            nn.ReLU(inplace=True)
            # nn.Softplus(beta=2.0)
        )

    def forward(self, x):
        return self.head(x)


class EdgeHead(nn.Module):
    def __init__(self, in_channels, mid_channels=None):
        super(EdgeHead, self).__init__()
        mid_channels = mid_channels or max(in_channels // 4, 16)
        self.refine = BasicConv2d(in_channels, mid_channels, kernel_size=1)
        self.pred = nn.Conv2d(mid_channels, 1, kernel_size=3, padding=1)

        nn.init.kaiming_normal_(self.pred.weight, mode='fan_out', nonlinearity='relu')
        if self.pred.bias is not None:
            nn.init.constant_(self.pred.bias, 0)

    def forward(self, x):
        x = self.refine(x)
        return self.pred(x)


class FusionBlock(nn.Module):
    def __init__(self, in_channels, num_input):
        super(FusionBlock, self).__init__()
        self.num_input = num_input
        self.conv_add = nn.ModuleList([
            BasicConv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            for _ in range(num_input - 1)
        ])
        self.resize = ResizeToMatch()
        self.head = FusionHead(in_channels)

    def forward(self, *features):
        if len(features) != self.num_input:
            raise ValueError(f"Expected {self.num_input} input features, got {len(features)}")

        feat = features[-1]

        for i in range(self.num_input - 2, -1, -1):
            target_feat = features[i]
            feat_aligned = self.resize(feat, target_feat)
            feat = self.conv_add[i](feat_aligned + target_feat)

        return self.head(feat)

class SIFM(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(SIFM, self).__init__()
        self.resize_match = ResizeToMatch()

        self.fconv11 = BasicConv2d(in_planes, in_planes, kernel_size=3, padding=1)
        self.fconv12 = BasicConv2d(in_planes, in_planes, kernel_size=3, padding=1)

        self.fconv21 = BasicConv2d(in_planes, in_planes, kernel_size=3, padding=1)
        self.fconv22 = BasicConv2d(in_planes, in_planes, kernel_size=3, padding=1)

        self.fconv31 = BasicConv2d(in_planes, out_planes, kernel_size=3, padding=1)
        self.fconv32 = BasicConv2d(in_planes, out_planes, kernel_size=3, padding=1)

        self.conv_add = BasicConv2d(out_planes, out_planes, kernel_size=3, padding=1)
        self.conv_edge = BasicConv2d(out_planes*2, out_planes, kernel_size=3, padding=1)

    def forward(self, x1, x2, edge=None):
        x2 = self.resize_match(x2, x1)

        y1 = x1 * x2
        y2 = x1 + x2

        y1 = self.fconv11(y1) + x1
        y2 = self.fconv12(y2) + x2

        f1 = self.fconv21(y1) * y2
        f2 = self.fconv22(y2) + y1

        out = self.fconv31(f1) + self.fconv32(f2)
        out = self.conv_add(out)

        if edge is not None:
            edge = self.resize_match(edge, out)
            out = self.conv_edge(torch.cat((out, edge), dim=1))

        return out

class MLFM(nn.Module):
    def __init__(self, channel, layers=4):
        super(MLFM, self).__init__()
        self.layers = layers

        self.sifm11 = SIFM(channel, channel)
        self.sifm12 = SIFM(channel, channel)
        self.sifm13 = SIFM(channel, channel)
        self.sifm14 = SIFM(channel, channel)
        self.fusion1 = FusionBlock(channel, 4)

        self.sifm21 = SIFM(channel, channel)
        self.sifm22 = SIFM(channel, channel)
        self.sifm23 = SIFM(channel, channel)
        self.fusion2 = FusionBlock(channel, 3)

        self.sifm31 = SIFM(channel, channel)
        self.sifm32 = SIFM(channel, channel)
        self.fusion3 = FusionBlock(channel, 2)

        self.sifm41 = SIFM(channel, channel)
        self.fusion4 = FusionBlock(channel, 1)

    def forward(self, x0, x1, x2, x3, x4, edge=None):
        evidences = []

        # First Layer
        if self.layers >= 1:
            x11 = self.sifm11(x0, x1, edge)
            x12 = self.sifm12(x1, x2, edge)
            x13 = self.sifm13(x2, x3, edge)
            x14 = self.sifm14(x3, x4, edge)

            evidences.append(self.fusion1(x11, x12, x13, x14))

        # Second Layer
        if self.layers >= 2:
            x21 = self.sifm21(x11, x12, edge)
            x22 = self.sifm22(x12, x13, edge)
            x23 = self.sifm23(x13, x14, edge)

            evidences.append(self.fusion2(x21, x22, x23))

        # Third Layer
        if self.layers >= 3:
            x31 = self.sifm31(x21, x22, edge)
            x32 = self.sifm32(x22, x23, edge)

            evidences.append(self.fusion3(x31, x32))

        # Fourth Layer
        if self.layers >= 4:
            x41 = self.sifm41(x31, x32, edge)
            evidences.append(self.fusion4(x41))

        return evidences

class EDM(nn.Module):
    def __init__(self, channels, reduce_channel):
        super(EDM, self).__init__()
        self.resize_match = ResizeToMatch()

        # 降维使用 3x3 卷积 (kernel_size=3, padding=1) 以提供空间感受野，抑制混叠效应
        self.r_conv4 = BasicConv2d(reduce_channel[4], channels, kernel_size=3, padding=1)
        self.r_conv3 = BasicConv2d(reduce_channel[3], channels, kernel_size=3, padding=1)
        self.r_conv2 = BasicConv2d(reduce_channel[2], channels, kernel_size=3, padding=1)
        self.r_conv1 = BasicConv2d(reduce_channel[1], channels, kernel_size=3, padding=1)

        self.conv_add4 = BasicConv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_add3 = BasicConv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_add2 = BasicConv2d(channels, channels, kernel_size=3, padding=1)
        self.conv_add1 = BasicConv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x1, x2, x3, x4):
        x4 = self.r_conv4(x4)
        x3 = self.r_conv3(x3)
        x2 = self.r_conv2(x2)
        x1 = self.r_conv1(x1)

        x4 = self.resize_match(x4, x3)
        x3 = self.conv_add4(x3 + x4)

        x3 = self.resize_match(x3, x2)
        x2 = self.conv_add3(x2 + x3)

        x2 = self.resize_match(x2, x1)
        edge = self.conv_add2(x1 + x2)

        return edge

class DynamicWeight(nn.Module):
    def __init__(self, enum):
        super(DynamicWeight, self).__init__()
        self.enum = enum

    def forward(self, *evidences):
        enum = self.enum
        weight_matrix = [torch.zeros_like(evidences[0][:,0,:,:]).unsqueeze(1).repeat(1,enum,1,1) for _ in range(enum)]

        with torch.no_grad():
            for i in range(enum):
                for j in range(i,enum):
                    if i != j:
                        weight_matrix[i][:,j,:,:] = 1 - self.BJS(evidences[i], evidences[j])

            row_sum = []
            for tensor in weight_matrix:
                row_sum.append(torch.sum(tensor, dim=1, keepdim=True))
            row_sum = torch.cat(row_sum, dim=1)

            col_sum = 0
            for tensor in weight_matrix:
                col_sum += tensor

            weight_matrix = (row_sum + col_sum) / (enum - 1)
            # weight_matrix = weight_matrix / (weight_matrix.sum(dim=1, keepdim=True) + 1e-6)


        return weight_matrix

    def BJS(self, m1, m2):
        m12 = (m1 + m2) / 2
        H1 = -torch.sum(m1 * torch.log(m1 + 1e-5), dim=1)
        H2 = -torch.sum(m2 * torch.log(m2 + 1e-5), dim=1)
        H12 = -torch.sum(m12 * torch.log(m12 + 1e-5), dim=1)
        return H12 - 0.5 * (H1 + H2)

class EUNet(nn.Module):
    def __init__(self, opt, channel=128):
        super(EUNet, self).__init__()
        self.backbone = build_backbone(opt)
        self.encoder = self.backbone.encoder
        self.transform_feat = self.backbone.feature_transform

        self.r_convs = nn.ModuleList([
            RFB(in_channel, channel) for in_channel in self.backbone.out_channels
        ])

        self.mlfm = MLFM(channel, layers=4)
        self.edm = EDM(channel, self.backbone.out_channels)
        self.conv_edge = EdgeHead(channel)
        self.weight = DynamicWeight(opt.enum)

    def forward(self, x):
        imsize = x.size()[2:]
        features = self.encoder(x)

        feat0 = self.transform_feat(features[0])
        feat1 = self.transform_feat(features[1])
        feat2 = self.transform_feat(features[2])
        feat3 = self.transform_feat(features[3])
        feat4 = self.transform_feat(features[4])

        edge_feat = self.edm(feat1, feat2, feat3, feat4)

        x0 = self.r_convs[0](feat0)
        x1 = self.r_convs[1](feat1)
        x2 = self.r_convs[2](feat2)
        x3 = self.r_convs[3](feat3)
        x4 = self.r_convs[4](feat4)

        branch_evidences = self.mlfm(x0, x1, x2, x3, x4, edge_feat)

        branch_evidences = [
            F.interpolate(ev, size=imsize, mode='bilinear', align_corners=False)
            for ev in branch_evidences
        ]

        smooth = 1e-4
        evidence_sum = 0
        masses = []
        for ev in branch_evidences:
            K_matrix = torch.full(
                (ev.shape[0], 1, *ev.shape[2:]),
                2,
                dtype=ev.dtype,
                device=ev.device
            )
            mass = torch.cat((ev, K_matrix), dim=1)
            mass = mass / (torch.sum(mass, dim=1, keepdim=True) + smooth)
            masses.append(mass)

        weight_matrix = self.weight(*masses)

        for i, ev in enumerate(branch_evidences):
            evidence_sum += weight_matrix[:, i, :, :].unsqueeze(1) * ev

        edge_feat = F.interpolate(edge_feat, size=imsize, mode='bilinear', align_corners=False)
        edge_out = self.conv_edge(edge_feat)

        return {
            "final_evidence": evidence_sum,
            "edge_pred": edge_out,
            "branch_evidence": branch_evidences,
            "branch_weights": weight_matrix
        }
