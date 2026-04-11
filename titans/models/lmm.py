"""
Titans-LMM  —  Standalone Long-Term Memory Model  (§4.3 / §5.1)
=================================================================
The neural memory module used *alone*, without any attention branch.
From the paper: "we expect our long-term memory to effectively learn
from data, even without attention."

Architecture (per transformer block):
    x  →  LayerNorm  →  NeuralMemory  →  +residual  →  FFN  →  +residual

A stack of `n_layers` such blocks is followed by a final LayerNorm and a
language-model head (linear projection to vocab_size).
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from titans.memory.neural_memory import NeuralMemory
from titans.memory.persistent_memory import PersistentMemory


class _LMMBlock(nn.Module):
    """Single Titans-LMM transformer block."""

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        mem_layers: int,
        chunk_size: int,
        ffn_mult: int,
        use_momentum: bool,
        use_decay: bool,
    ):
        super().__init__()
        self.norm1  = nn.LayerNorm(d_model)
        self.memory = NeuralMemory(
            d_model      = d_model,
            d_hidden     = d_hidden,
            n_layers     = mem_layers,
            chunk_size   = chunk_size,
            use_momentum = use_momentum,
            use_decay    = use_decay,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.SiLU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:          # (B, T, D)
        x = x + self.memory(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class TitansLMM(nn.Module):
    """
    Titans — LMM-only variant (no attention).

    Parameters
    ----------
    vocab_size   : int   — vocabulary / input dimension
    d_model      : int   — model hidden dimension
    n_layers     : int   — number of LMM blocks
    mem_layers   : int   — depth of each memory MLP (L_M)
    d_hidden     : int   — hidden dim of memory MLP (default: 4*d_model)
    n_persistent : int   — number of persistent-memory tokens (0 = disable)
    chunk_size   : int   — chunk size for parallel inner-loop training
    ffn_mult     : int   — FFN hidden-dim multiplier
    max_seq_len  : int   — maximum sequence length (for positional embedding)
    use_momentum : bool
    use_decay    : bool
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int      = 512,
        n_layers: int     = 12,
        mem_layers: int   = 2,
        d_hidden: Optional[int] = None,
        n_persistent: int = 16,
        chunk_size: int   = 64,
        ffn_mult: int     = 4,
        max_seq_len: int  = 8192,
        use_momentum: bool = True,
        use_decay: bool    = True,
    ):
        super().__init__()
        self.d_model      = d_model
        self.n_persistent = n_persistent

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len + n_persistent, d_model)

        # Optional persistent memory
        self.persistent = (
            PersistentMemory(n_persistent, d_model) if n_persistent > 0 else None
        )

        self.blocks = nn.ModuleList([
            _LMMBlock(
                d_model      = d_model,
                d_hidden     = d_hidden or (4 * d_model),
                mem_layers   = mem_layers,
                chunk_size   = chunk_size,
                ffn_mult     = ffn_mult,
                use_momentum = use_momentum,
                use_decay    = use_decay,
            )
            for _ in range(n_layers)
        ])

        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,                  # (B, T)  integer token ids
        labels: Optional[Tensor] = None,    # (B, T)  for computing LM loss
    ) -> dict:
        """
        Parameters
        ----------
        input_ids : (B, T)
        labels    : (B, T) optional; if given, cross-entropy loss is returned.

        Returns
        -------
        dict with keys:
            'logits' : (B, T, vocab_size)
            'loss'   : scalar Tensor  (only when labels is not None)
        """
        B, T = input_ids.shape
        device = input_ids.device

        x = self.tok_emb(input_ids)                            # (B, T, D)

        # Prepend persistent memory
        if self.persistent is not None:
            x = self.persistent(x)                             # (B, Np+T, D)

        T_aug = x.size(1)
        pos   = torch.arange(T_aug, device=device)
        x     = x + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Strip persistent prefix before computing logits
        if self.persistent is not None:
            x = self.persistent.strip(x)                       # (B, T, D)

        logits = self.lm_head(x)                               # (B, T, V)

        out: dict = {"logits": logits}
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        return out

    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float  = 1.0,
        top_k: int          = 50,
    ) -> Tensor:
        """Simple greedy / top-k sampler for quick evaluation."""
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)["logits"]         # (B, T, V)
            next_logits = logits[:, -1, :] / temperature       # (B, V)
            if top_k > 0:
                v, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < v[:, [-1]]] = float("-inf")
            probs   = torch.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
