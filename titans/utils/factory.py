"""
Model factory — instantiate any Titans variant from a TitansConfig.
"""

from __future__ import annotations
import torch.nn as nn
from titans.utils.config import TitansConfig


def build_model(cfg: TitansConfig) -> nn.Module:
    """
    Instantiate a Titans model from a TitansConfig.

    Parameters
    ----------
    cfg : TitansConfig

    Returns
    -------
    nn.Module — one of TitansMAC / TitansMAG / TitansMAL / TitansLMM
    """
    shared = dict(
        vocab_size   = cfg.vocab_size,
        d_model      = cfg.d_model,
        n_layers     = cfg.n_layers,
        mem_layers   = cfg.mem_layers,
        d_hidden     = cfg.d_hidden,
        n_persistent = cfg.n_persistent,
        chunk_size   = cfg.chunk_size,
        ffn_mult     = cfg.ffn_mult,
        max_seq_len  = cfg.max_seq_len,
        use_momentum = cfg.use_momentum,
        use_decay    = cfg.use_decay,
    )

    v = cfg.variant.upper()

    if v == "MAC":
        from titans.models.mac import TitansMAC
        return TitansMAC(**shared)

    if v == "MAG":
        from titans.models.mag import TitansMAG
        return TitansMAG(**shared, window=cfg.window)

    if v == "MAL":
        from titans.models.mal import TitansMAL
        return TitansMAL(**shared, window=cfg.window)

    if v == "LMM":
        from titans.models.lmm import TitansLMM
        return TitansLMM(**shared)

    raise ValueError(
        f"Unknown variant '{cfg.variant}'. "
        f"Choose one of: 'MAC', 'MAG', 'MAL', 'LMM'."
    )
