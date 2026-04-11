"""
Associative Scan
================
Solves the first-order linear recurrence element-wise:

    S_t = eta_t * S_{t-1} + u_t          (B, T, D)

Used by NeuralMemory to parallelise the momentum hidden-state computation.
The sequential implementation is always numerically exact and is accelerated
by torch.compile() via loop-fusion on modern hardware.
"""

from __future__ import annotations
import torch
from torch import Tensor


def _scan_sequential(eta: Tensor, u: Tensor) -> Tensor:
    """Sequential scan — O(T) numerically exact reference."""
    B, T, D = u.shape
    S       = torch.zeros(B, D, dtype=u.dtype, device=u.device)
    out     = torch.empty_like(u)
    for t in range(T):
        S        = eta[:, t, :] * S + u[:, t, :]
        out[:, t, :] = S
    return out                             # (B, T, D)


def parallel_scan(
    eta: Tensor,
    u: Tensor,
    use_parallel: bool = True,             # retained for API compatibility
) -> Tensor:
    """
    Solve  S_t = eta_t * S_{t-1} + u_t  across the time dimension.

    Parameters
    ----------
    eta          : (B, T, D)  data-dependent decay in [0, 1]
    u            : (B, T, D)  innovation / input at each step
    use_parallel : bool       ignored (kept for call-site compatibility)

    Returns
    -------
    S : (B, T, D)
    """
    return _scan_sequential(eta, u)
