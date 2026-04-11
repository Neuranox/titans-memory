"""
Persistent Memory Module (§3.3)
================================
Learnable but *input-independent* parameters prepended to every sequence.

Motivation (three perspectives from the paper):
  1. Memory perspective  – encodes task-level knowledge (how to do the task),
     separate from contextual memories that depend on the input.
  2. FFN perspective     – equivalent to the data-independent keys/values in a
     Feed-Forward Network interpreted as an associative memory.
  3. Technical           – mitigates attention-sink bias toward the first token
     by redistributing attention weights (Xiao et al. 2024; Han et al. 2024).

Usage
-----
    pm = PersistentMemory(n_tokens=16, d_model=512)
    x_augmented = pm(x)          # prepend P to sequence
    x_out        = pm.strip(y)   # remove the first n_tokens from output
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor


class PersistentMemory(nn.Module):
    """
    Persistent (task) memory: a set of N_p learnable token embeddings that
    are prepended to every input sequence before the attention module.

    Parameters
    ----------
    n_tokens : int   — N_p, number of persistent-memory tokens (≥ 1)
    d_model  : int   — embedding dimension
    """

    def __init__(self, n_tokens: int, d_model: int):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model  = d_model
        # P = [p_1, p_2, ..., p_{N_p}]   (eq. 19)
        self.P = nn.Parameter(torch.empty(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.P, std=0.02)

    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Prepend persistent-memory tokens to the input sequence.

        Parameters
        ----------
        x : (B, T, d_model)

        Returns
        -------
        x_new : (B, N_p + T, d_model)   — [P || x]  (eq. 19 / 22 / 26 / 29)
        """
        B = x.size(0)
        p = self.P.expand(B, -1, -1)           # (B, N_p, D)
        return torch.cat([p, x], dim=1)         # (B, N_p + T, D)

    def strip(self, y: Tensor) -> Tensor:
        """
        Remove the persistent-memory prefix from an output sequence.

        Parameters
        ----------
        y : (B, N_p + T, d_model)

        Returns
        -------
        out : (B, T, d_model)
        """
        return y[:, self.n_tokens:, :]

    # ------------------------------------------------------------------
    # Convenience: keep the persistent tokens frozen during test-time
    # memory updates (they encode task knowledge, not context)
    # ------------------------------------------------------------------

    def freeze(self):
        """Freeze persistent parameters (e.g. at test time)."""
        self.P.requires_grad_(False)

    def unfreeze(self):
        """Unfreeze persistent parameters for training."""
        self.P.requires_grad_(True)

    def extra_repr(self) -> str:
        return f"n_tokens={self.n_tokens}, d_model={self.d_model}"
