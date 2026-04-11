"""
Titans-MAG  —  Memory as a Gate  (§4.2)
=========================================
Two parallel branches combined by a learnable gating mechanism:
  (1) Core branch  — Sliding-Window Attention (SWA) with persistent prefix
  (2) Memory branch — NeuralMemory applied to the full (prefix-augmented) input

Output (eq. 26-28):
    x̃   = [P || x]
    y    = SW-Attn*(x̃)
    o    = y  ⊗  M(x̃)          where ⊗ is a learned sigmoid gate

The gating normalises both branches with learnable scale vectors and
then applies σ(.) before multiplying, as described in §4.2.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from titans.memory.neural_memory import NeuralMemory
from titans.memory.persistent_memory import PersistentMemory
from titans.ops.attention import sliding_window_attention


# ---------------------------------------------------------------------------
# Single MAG block
# ---------------------------------------------------------------------------

class _MAGBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        mem_layers: int,
        chunk_size: int,
        n_persistent: int,
        window: int,
        ffn_mult: int,
        use_momentum: bool,
        use_decay: bool,
    ):
        super().__init__()
        self.n_persistent = n_persistent
        self.window       = window

        # ---- (1) Persistent memory ----
        self.persistent = (
            PersistentMemory(n_persistent, d_model) if n_persistent > 0 else None
        )

        # ---- (2) Sliding-window attention branch ----
        self.norm_attn = nn.LayerNorm(d_model)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        # ---- (3) Neural memory branch ----
        self.norm_mem = nn.LayerNorm(d_model)
        self.long_mem = NeuralMemory(
            d_model      = d_model,
            d_hidden     = d_hidden,
            n_layers     = mem_layers,
            chunk_size   = chunk_size,
            use_momentum = use_momentum,
            use_decay    = use_decay,
        )

        # ---- (4) Gating ----
        # Learnable scale vectors for each branch (per eq. §4.2 description)
        self.scale_attn = nn.Parameter(torch.ones(d_model))
        self.scale_mem  = nn.Parameter(torch.ones(d_model))
        self.gate_proj  = nn.Linear(d_model, d_model, bias=True)

        # ---- (5) Output projection + FFN ----
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm_out = nn.LayerNorm(d_model)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn      = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.SiLU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    # -----------------------------------------------------------------------

    def _swa(self, x: Tensor) -> Tensor:
        """Sliding-window attention."""
        B, T, D = x.shape
        q = F.normalize(self.W_Q(x), dim=-1) * (D ** 0.5)
        k = F.normalize(self.W_K(x), dim=-1)
        v = self.W_V(x)
        out = sliding_window_attention(q, k, v,
                                       window   = self.window,
                                       n_prefix = self.n_persistent)
        return self.attn_out(out)

    # -----------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """x : (B, T, D)"""
        # (1) Prepend persistent tokens: x̃ = [P || x]
        if self.persistent is not None:
            x_aug = self.persistent(x)                       # (B, Np+T, D)
        else:
            x_aug = x

        # (2) SWA branch
        y_attn = x_aug + self._swa(self.norm_attn(x_aug))   # (B, Np+T, D)

        # (3) Memory branch
        y_mem  = self.long_mem(self.norm_mem(x_aug))         # (B, Np+T, D)

        # (4) Normalise + gate  (§4.2)
        y_attn_n = y_attn * self.scale_attn
        y_mem_n  = y_mem  * self.scale_mem
        gate     = torch.sigmoid(self.gate_proj(y_attn_n))
        combined = gate * y_attn_n + (1.0 - gate) * y_mem_n # (B, Np+T, D)
        combined = self.out_proj(combined)

        # Strip persistent prefix
        if self.persistent is not None:
            combined = self.persistent.strip(combined)       # (B, T, D)

        # Residual + FFN
        out = x + self.norm_out(combined)
        out = out + self.ffn(self.norm_ffn(out))
        return out


# ---------------------------------------------------------------------------
# Full TitansMAG model
# ---------------------------------------------------------------------------

class TitansMAG(nn.Module):
    """
    Titans — Memory as a Gate (MAG).

    Parameters
    ----------
    vocab_size   : int
    d_model      : int
    n_layers     : int
    mem_layers   : int   — depth L_M of each NeuralMemory MLP
    d_hidden     : int
    n_persistent : int   — N_p persistent-memory tokens
    window       : int   — SWA local window size W
    chunk_size   : int   — inner-loop chunk size
    ffn_mult     : int
    max_seq_len  : int
    use_momentum : bool
    use_decay    : bool
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int   = 512,
        n_layers: int  = 12,
        mem_layers: int = 2,
        d_hidden: Optional[int] = None,
        n_persistent: int = 16,
        window: int       = 512,
        chunk_size: int   = 64,
        ffn_mult: int     = 4,
        max_seq_len: int  = 8192,
        use_momentum: bool = True,
        use_decay: bool    = True,
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            _MAGBlock(
                d_model      = d_model,
                d_hidden     = d_hidden or (4 * d_model),
                mem_layers   = mem_layers,
                chunk_size   = chunk_size,
                n_persistent = n_persistent,
                window       = window,
                ffn_mult     = ffn_mult,
                use_momentum = use_momentum,
                use_decay    = use_decay,
            )
            for _ in range(n_layers)
        ])

        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
    ) -> dict:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        x    = self.tok_emb(input_ids) + self.pos_emb(pos)

        for block in self.blocks:
            x = block(x)

        x      = self.norm(x)
        logits = self.lm_head(x)

        out: dict = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.size(-1)),
                labels[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )
            out["loss"] = loss
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 50,
        temperature: float  = 1.0,
        top_k: int          = 50,
    ) -> Tensor:
        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)["logits"]
            next_l = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_l, top_k)
                next_l[next_l < v[:, [-1]]] = float("-inf")
            probs   = torch.softmax(next_l, dim=-1)
            next_id = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_id], dim=1)
        return input_ids
