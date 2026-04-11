"""
Attention Primitives
====================
Pure-PyTorch implementations of:
  • causal_attention        – standard full causal self-attention (O(N²))
  • sliding_window_attention – local causal attention with window size W

Both accept an optional prefix of persistent-memory tokens that always attend
to every other token (the prefix is never masked out in the *query* direction).
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor


def causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
    n_prefix: int = 0,
) -> Tensor:
    """
    Standard scaled-dot-product causal attention.

    Parameters
    ----------
    q, k, v  : (B, T, D)
    mask     : optional boolean mask (B, T, T) – True means *keep*
    n_prefix : number of persistent-memory tokens prepended to the sequence;
               these are never masked away from the keys/values side.

    Returns
    -------
    out : (B, T, D)
    """
    B, T, D = q.shape
    scale = math.sqrt(D)

    # Build causal mask: lower-triangular (token i can attend to j ≤ i)
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()

    # Persistent-memory prefix: all positions can attend to the first n_prefix slots
    if n_prefix > 0:
        causal[:, :n_prefix] = True

    if mask is not None:
        causal = causal & mask

    attn_bias = torch.zeros(T, T, dtype=q.dtype, device=q.device)
    attn_bias = attn_bias.masked_fill(~causal, float("-inf"))

    scores = torch.bmm(q, k.transpose(-2, -1)) / scale  # (B, T, T)
    scores = scores + attn_bias.unsqueeze(0)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, v)


def sliding_window_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    window: int = 512,
    n_prefix: int = 0,
) -> Tensor:
    """
    Causal sliding-window attention (SWA).

    Token i attends only to tokens in [i - window + 1, i] plus the
    persistent-memory prefix tokens [0, n_prefix).

    Parameters
    ----------
    q, k, v  : (B, T, D)
    window   : local context size W
    n_prefix : number of persistent-memory tokens (always visible)

    Returns
    -------
    out : (B, T, D)
    """
    B, T, D = q.shape
    scale = math.sqrt(D)

    # Lower-triangular causal mask
    causal = torch.ones(T, T, dtype=torch.bool, device=q.device).tril()

    # Keep only last `window` positions (band mask)
    rows = torch.arange(T, device=q.device).unsqueeze(1)  # (T, 1)
    cols = torch.arange(T, device=q.device).unsqueeze(0)  # (1, T)
    band = (rows - cols) < window
    causal = causal & band

    # Persistent-memory prefix always visible
    if n_prefix > 0:
        causal[:, :n_prefix] = True

    attn_bias = torch.zeros(T, T, dtype=q.dtype, device=q.device)
    attn_bias = attn_bias.masked_fill(~causal, float("-inf"))

    scores = torch.bmm(q, k.transpose(-2, -1)) / scale
    scores = scores + attn_bias.unsqueeze(0)
    weights = F.softmax(scores, dim=-1)
    return torch.bmm(weights, v)
