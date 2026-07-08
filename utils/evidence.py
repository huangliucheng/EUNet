import torch


def _cfg_get(cfg, name, default=None):
    if cfg is None:
        return default
    return getattr(cfg, name, default)


def evidence_to_belief(evidence):
    return evidence / torch.sum(evidence + 1, dim=1, keepdim=True)


def evidence_to_probability(evidence):
    alpha = evidence + 1
    return alpha / (torch.sum(alpha, dim=1, keepdim=True) + 1e-6)


def evidence_to_uncertainty(evidence):
    strength = torch.sum(evidence + 1, dim=1, keepdim=True)
    return 2.0 / (strength + 1e-6)


def evidence_to_prediction(evidence, cfg=None):
    inference_cfg = _cfg_get(cfg, "inference")
    output_mode = _cfg_get(inference_cfg, "output_mode", "foreground_belief")
    if output_mode == "projected_probability":
        return evidence_to_probability(evidence)[:, 0:1, :, :]
    if output_mode == "foreground_belief":
        return evidence_to_belief(evidence)[:, 0:1, :, :]
    raise ValueError(f"Unsupported inference.output_mode: {output_mode}")
