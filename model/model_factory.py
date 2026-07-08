from .EUNet import EUNet


def build_model(cfg):
    return EUNet(cfg, cfg.unified_channel)
