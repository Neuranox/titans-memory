"""
Titans-MAL  —  Memory as a Layer  (§4.3)
==========================================
Stacks NeuralMemory and Sliding-Window Attention as sequential layers
(eq. 29-31):
    x̃   = [P || x]
    y    = M(x̃)         — memory compresses context
    o    = SW-Attn(y)   — attention refines compressed representation

This is the most common hybrid design in the literature (similar to H3,
Mamba+Attn, Samba), and is included for direct comparison with those baselines.
Each transformer block alternates between LMM and SWA sub-layers.
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
# Single MAL block
# ---------------------------------------------------------------------------

class _MALBlock(nn.Module):

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

        # ---- Persistent memory ----
        self.persistent = (
            PersistentMemory(n_persistent, d_model) if n_persistent > 0 else None
        )

        # ---- Sub-layer 1: NeuralMemory (eq. 30) ----
        self.norm_mem = nn.LayerNorm(d_model)
        self.long_mem = NeuralMemory(
            d_model      = d_model,
            d_hidden     = d_hidden,
            n_layers     = mem_layers,
            chunk_size   = chunk_size,
            use_momentum = use_momentum,
            use_decay    = use_decay,
        )

        # ---- Sub-layer 2: Sliding-window attention (eq. 31) ----
        self.norm_attn = nn.LayerNorm(d_model)
        self.W_Q   = nn.Linear(d_model, d_model, bias=False)
        self.W_K   = nn.Linear(d_model, d_model, bias=False)
        self.W_V   = nn.Linear(d_model, d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        # ---- FFN ----
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn      = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.SiLU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    # -----------------------------------------------------------------------

    def _swa(self, x: Tensor) -> Tensor:
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
        # Prepend persistent tokens: x̃ = [P || x]
        if self.persistent is not None:
            x_aug = self.persistent(x)                        # (B, Np+T, D)
        else:
            x_aug = x

        # Sub-layer 1 – memory (eq. 30): y = M(x̃)
        y = x_aug + self.long_mem(self.norm_mem(x_aug))       # (B, Np+T, D)

        # Sub-layer 2 – SWA (eq. 31): o = SW-Attn(y)
        o = y + self._swa(self.norm_attn(y))                  # (B, Np+T, D)

        # Strip persistent prefix
        if self.persistent is not None:
            o = self.persistent.strip(o)                      # (B, T, D)

        # FFN
        out = o + self.ffn(self.norm_ffn(o))
        return out


# ---------------------------------------------------------------------------
# Full TitansMAL model
# ---------------------------------------------------------------------------

class TitansMAL(nn.Module):
    """
    Titans — Memory as a Layer (MAL).

    Parameters
    ----------
    vocab_size   : int
    d_model      : int
    n_layers     : int
    mem_layers   : int   — depth L_M of each NeuralMemory MLP
    d_hidden     : int
    n_persistent : int   — N_p persistent-memory tokens
    window       : int   — SWA local window size W
    chunk_size   : int
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
            _MALBlock(
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
