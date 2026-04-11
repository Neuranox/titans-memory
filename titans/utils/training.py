"""
Training utilities
==================
Helpers used in the training loop that are otherwise annoying to re-implement.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return total (or trainable-only) parameter count."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """
    Cosine annealing with linear warm-up, as used in the Titans paper
    (following LLaMA / Gated DeltaNet training procedure).

    Parameters
    ----------
    optimizer     : torch.optim.Optimizer
    warmup_steps  : int   — linear ramp duration
    total_steps   : int   — total training steps
    min_lr_ratio  : float — lr_min / lr_peak  (default: 0.1 → min = 10% peak)
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)


def build_optimizer(
    model: nn.Module,
    lr: float           = 4e-4,
    weight_decay: float = 0.1,
    beta1: float        = 0.9,
    beta2: float        = 0.95,
) -> AdamW:
    """
    AdamW with weight-decay applied only to non-bias / non-norm parameters
    (standard best practice, matching paper training setup).
    """
    decay     = []
    no_decay  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    groups = [
        {"params": decay,    "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=lr, betas=(beta1, beta2))
