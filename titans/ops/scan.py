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


def _scan_parallel_log(eta: Tensor, u: Tensor) -> Tensor:
    """Truly parallel scan using log-space cumsum (O(log T))."""
    # S_t = sum_{i=0}^t u_i * exp(cumsum(log(eta))_t - cumsum(log(eta))_i)
    # This is much faster on GPUs for long sequences.
    
    # Add epsilon for numerical stability in log
    log_eta = torch.log(eta.clamp(min=1e-8))
    log_eta_cumsum = torch.cumsum(log_eta, dim=1)
    
    # We use the identity: exp(A-B) = exp(A)/exp(B)
    # S_t = exp(cumsum_t) * sum_{i=0}^t (u_i / exp(cumsum_i))
    phi = torch.exp(log_eta_cumsum)
    epsilon = 1e-12
    innovation = u / (phi + epsilon)
    cumulative_innovation = torch.cumsum(innovation, dim=1)
    
    return phi * cumulative_innovation


def parallel_scan(
    eta: Tensor,
    u: Tensor,
    use_parallel: bool = True,
) -> Tensor:
    """
    High-performance scan interface.
    """
    if use_parallel and eta.is_cuda:
        try:
            return _scan_parallel_log(eta, u)
        except Exception:
            return _scan_sequential(eta, u)
    return _scan_sequential(eta, u)
