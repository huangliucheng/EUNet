import torch
import torch.nn as nn
import torch.nn.functional as F


MAIN_EDL_WEIGHT = 1.0
MAIN_KL_WEIGHT = 1.0

# Hard-coded auxiliary branch controls. Set per branch in MLFM output order.
# 0.0 means the branch keeps BCE+Dice supervision but does not use that EDL/KL term.
BRANCH_EDL_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
BRANCH_KL_WEIGHTS = (1.0, 1.0, 1.0, 1.0)


def _cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def _cfg_sequence(cfg, name, default):
    value = _cfg_get(cfg, name, default)
    if value is None:
        return default
    return value


def focal_loss(inputs, targets, alpha=0.5, gamma=1.5, reduction='mean'):
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # pt = p if target = 1 else 1-p
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduction == 'mean':
        return F_loss.mean()
    elif reduction == 'sum':
        return F_loss.sum()
    else:
        return F_loss

# (备用) 纯边缘 BCE 损失
def bce_edge_loss(edge_pred, edge):
    num_pos = edge.sum()
    num_neg = edge.numel() - num_pos
    pos_weight = (num_neg / (num_pos + 1e-6)).clamp(max=20.0)
    loss_bce = F.binary_cross_entropy_with_logits(edge_pred, edge, pos_weight=pos_weight)
    return loss_bce

def edge_dice_loss(pred, mask):
    smooth = 1e-4
    prob = torch.sigmoid(pred)
    intersection = torch.sum(prob * mask, dim=(2, 3))
    union = torch.sum(prob, dim=(2, 3)) + torch.sum(mask, dim=(2, 3))
    loss = 1 - (2 * intersection + smooth) / (union + smooth)
    return loss.mean()


def bce_loss(evidence, mask):
    mask = torch.cat((mask, 1-mask), dim=1)
    S = torch.sum(evidence + 1, dim=1, keepdim=True)
    belief = evidence / S
    loss = - torch.log(belief + 1e-5)* mask

    return loss.sum(dim=1).mean()

def dice_loss(evidence, mask):
    smooth = 1e-4
    S = torch.sum(evidence + 1, dim=1, keepdim=True)
    belief_foreground = evidence[:, 0:1, :, :] / S
    intersection = torch.sum(belief_foreground * mask, dim=(2, 3))
    union = torch.sum(belief_foreground, dim=(2, 3)) + torch.sum(mask, dim=(2, 3))
    loss = 1 - (2 * intersection + smooth) / (union + smooth)
    return loss.mean()


def kl_divergence(evidence, gt):
    alpha = evidence + 1
    alpha = gt + (1 - gt) * alpha

    two = torch.tensor(2.0, dtype=torch.float32, device=evidence.device)

    A = (
        torch.lgamma(torch.sum(alpha, dim=1, keepdim=True))
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(two)
    )
    B = ((alpha - 1) * (torch.digamma(alpha) - torch.digamma(alpha.sum(dim=1, keepdim=True)))).sum(dim=1, keepdim=True)
    return (A + B).squeeze(1)

def dirc_loss(
    evidences,
    mask,
    epoch,
    warmup_epochs=0,
    max_epochs=100,
    kl_annealing_epochs=None,
    edl_weight=1.0,
    kl_weight=1.0,
):
    if kl_annealing_epochs is None:
        annealing_epochs = max_epochs // 2
    else:
        annealing_epochs = int(kl_annealing_epochs)
        if annealing_epochs <= 0:
            raise ValueError("loss.kl_annealing_epochs must be a positive integer")

    if epoch < warmup_epochs:
        annealing_coef = 0.0
    else:
        annealing_coef = min(1.0, (epoch - warmup_epochs) / annealing_epochs)

    alpha = evidences + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    pred = alpha / S
    mask = torch.cat((mask, 1 - mask), dim=1)

    kl_div = kl_divergence(evidences, mask)
    edl_term = torch.sum((mask - pred)**2 + pred * (1 - pred) / (S + 1), dim=1)
    loss = edl_weight * edl_term + annealing_coef * kl_weight * kl_div
    return loss.mean()


def calc_edl_combined_loss(
    evidence,
    label,
    epoch,
    warmup_epochs=0,
    max_epochs=100,
    kl_annealing_epochs=None,
    edl_weight=1.0,
    kl_weight=1.0,
    use_edl=True,
):
    loss = dice_loss(evidence, label) + bce_loss(evidence, label)
    if use_edl:
        loss = loss + dirc_loss(
            evidence,
            label,
            epoch,
            warmup_epochs,
            max_epochs,
            kl_annealing_epochs=kl_annealing_epochs,
            edl_weight=edl_weight,
            kl_weight=kl_weight,
        )
    return loss

def multi_loss_function(preds_dict, label, edge, epoch, warmup_epochs=0, max_epochs=100, loss_cfg=None):

    final_evidence = preds_dict['final_evidence']
    edge_pred = preds_dict['edge_pred']
    branch_evidences = preds_dict['branch_evidence'] # list of [ev0, ev1, ev2, ev3]

    loss_dict = {}
    use_edge_loss = bool(_cfg_get(loss_cfg, "use_edge_loss", True))
    use_main_edl = bool(_cfg_get(loss_cfg, "use_main_edl", True))
    use_branch_loss = bool(_cfg_get(loss_cfg, "use_branch_loss", True))
    use_branch_edl = bool(_cfg_get(loss_cfg, "use_branch_edl", True))
    edge_loss_weight = float(_cfg_get(loss_cfg, "edge_loss_weight", 1.0))
    branch_loss_weight = float(_cfg_get(loss_cfg, "branch_loss_weight", 1.0))
    main_edl_weight = float(_cfg_get(loss_cfg, "main_edl_weight", MAIN_EDL_WEIGHT))
    main_kl_weight = float(_cfg_get(loss_cfg, "main_kl_weight", MAIN_KL_WEIGHT))
    branch_edl_weights = _cfg_sequence(loss_cfg, "branch_edl_weights", BRANCH_EDL_WEIGHTS)
    branch_kl_weights = _cfg_sequence(loss_cfg, "branch_kl_weights", BRANCH_KL_WEIGHTS)
    kl_annealing_epochs = _cfg_get(loss_cfg, "kl_annealing_epochs", None)

    if use_edge_loss and edge_pred is not None:
        loss_edge_bce = F.binary_cross_entropy_with_logits(edge_pred, edge)
        loss_edge_focal = focal_loss(edge_pred, edge)

        loss_edge = edge_loss_weight * (loss_edge_bce + loss_edge_focal)
        loss_dict['Loss_edge'] = loss_edge.item()
        loss_dict['Loss_edge_bce'] = loss_edge_bce.item()
        loss_dict['Loss_edge_focal'] = loss_edge_focal.item()
    else:
        loss_edge = 0

    loss_main = calc_edl_combined_loss(
        final_evidence,
        label,
        epoch,
        warmup_epochs,
        max_epochs,
        kl_annealing_epochs=kl_annealing_epochs,
        edl_weight=main_edl_weight,
        kl_weight=main_kl_weight,
        use_edl=use_main_edl,
    )
    loss_dict['Loss_main'] = loss_main.item()

    total_loss = loss_main + loss_edge
    if use_branch_loss:
        for i, branch_evidence in enumerate(branch_evidences):
            branch_edl_weight = branch_edl_weights[i] if i < len(branch_edl_weights) else branch_edl_weights[-1]
            branch_kl_weight = branch_kl_weights[i] if i < len(branch_kl_weights) else branch_kl_weights[-1]
            l_branch = calc_edl_combined_loss(
                branch_evidence,
                label,
                epoch,
                warmup_epochs,
                max_epochs,
                kl_annealing_epochs=kl_annealing_epochs,
                edl_weight=float(branch_edl_weight),
                kl_weight=float(branch_kl_weight),
                use_edl=use_branch_edl,
            )
            total_loss += branch_loss_weight * l_branch
            # 记录每个辅助特征的 Loss (如 Loss_aux_0, Loss_aux_1)
            loss_dict[f'Loss_aux_{i}'] = l_branch.item()

    loss_dict['Loss_total'] = total_loss.item()

    return total_loss, loss_dict
