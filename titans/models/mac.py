"""
Titans-MAC  —  Memory as a Context  (§4.1)
===========================================
Three-branch architecture:
  (1) Core branch        — full causal attention on [P || h_t || S^(t)]
  (2) Long-term memory   — NeuralMemory that learns at test time
  (3) Persistent memory  — task-knowledge tokens P (input-independent)

Per-segment processing  (Figure 2 / eq. 21-25):
    Given segment S^(t):
        q_t      = S^(t) @ W_Q
        h_t      = M*_{t-1}(q_t)              # retrieve from long-term mem
        S̃^(t)   = [P || h_t || S^(t)]         # augmented context
        y_t      = Attention(S̃^(t))            # full causal attn over window
        M_t      = M_{t-1}.update(y_t)         # write updated ctx into memory
        o_t      = y_t  ⊗  M*_t(y_t)          # gated output (eq. 25)
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from titans.memory.neural_memory import NeuralMemory
from titans.memory.persistent_memory import PersistentMemory
from titans.ops.attention import causal_attention


# ---------------------------------------------------------------------------
# Single MAC block
# ---------------------------------------------------------------------------

class _MACBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        d_hidden: int,
        mem_layers: int,
        chunk_size: int,
        n_persistent: int,
        ffn_mult: int,
        use_momentum: bool,
        use_decay: bool,
    ):
        super().__init__()
        self.n_persistent = n_persistent
        self.chunk_size   = chunk_size

        # ---- Long-term memory ----
        self.long_mem = NeuralMemory(
            d_model      = d_model,
            d_hidden     = d_hidden,
            n_layers     = mem_layers,
            chunk_size   = chunk_size,
            use_momentum = use_momentum,
            use_decay    = use_decay,
        )

        # ---- Persistent memory ----
        self.persistent = (
            PersistentMemory(n_persistent, d_model) if n_persistent > 0 else None
        )

        # ---- Attention projections ----
        self.W_Q_ctx = nn.Linear(d_model, d_model, bias=False)  # query for retrieval
        self.W_Q     = nn.Linear(d_model, d_model, bias=False)
        self.W_K     = nn.Linear(d_model, d_model, bias=False)
        self.W_V     = nn.Linear(d_model, d_model, bias=False)
        self.attn_out= nn.Linear(d_model, d_model, bias=False)

        # ---- Gating for final output (eq. 25) ----
        self.gate_norm = nn.LayerNorm(d_model)
        self.gate_proj = nn.Linear(d_model, d_model, bias=False)

        # ---- Norms & FFN ----
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn   = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.SiLU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )

    # -----------------------------------------------------------------------

    def _attention_on_segment(self, seq: Tensor) -> Tensor:
        """Full causal attention over the augmented sequence."""
        B, T, D = seq.shape
        q = F.normalize(self.W_Q(seq), dim=-1) * (D ** 0.5)
        k = F.normalize(self.W_K(seq), dim=-1)
        v = self.W_V(seq)
        out = causal_attention(q, k, v, n_prefix=self.n_persistent)
        return self.attn_out(out)

    # -----------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        x : (B, T, D)  — full sequence; split internally into segments.
        """
        B, T, D = x.shape
        C       = self.chunk_size
        outputs = []

        for start in range(0, T, C):
            end  = min(start + C, T)
            seg  = x[:, start:end, :]              # S^(t):  (B, c, D)

            # ---- (1) Retrieve from long-term memory (eq. 21) ----
            q_ctx = F.normalize(self.W_Q_ctx(seg), dim=-1)
            h_t   = self.long_mem.retrieve(q_ctx)  # (B, c, D)

            # ---- (2) Build augmented context (eq. 22) ----
            if self.persistent is not None:
                aug = self.persistent(torch.cat([h_t, seg], dim=1))
                # aug = [P || h_t || seg]
            else:
                aug = torch.cat([h_t, seg], dim=1)

            # ---- (3) Attention over augmented window (eq. 23) ----
            y_aug  = self.norm1(aug)
            y_aug  = aug + self._attention_on_segment(y_aug)

            # Strip prefix tokens to get y_t aligned with seg length
            if self.persistent is not None:
                y_t = self.persistent.strip(y_aug)   # (B, c+c_h, D) → (B, c+c_h, D)
            else:
                y_t = y_aug
            # Take only last `c` tokens (segment output, not h_t)
            y_t = y_t[:, -seg.size(1):, :]           # (B, c, D)

            # ---- (4) Update long-term memory with y_t (eq. 24) ----
            _ = self.long_mem(y_t)                    # runs write-pass

            # ---- (5) Gated output (eq. 25) ----
            mem_out = self.long_mem.retrieve(F.normalize(self.W_Q_ctx(y_t), dim=-1))
            gate    = torch.sigmoid(self.gate_proj(self.gate_norm(y_t)))
            o_t     = y_t * gate + mem_out * (1.0 - gate)

            # FFN with residual
            o_t = o_t + self.ffn(self.norm2(o_t))
            outputs.append(o_t)

        return torch.cat(outputs, dim=1)              # (B, T, D)


# ---------------------------------------------------------------------------
# Full TitansMAC model
# ---------------------------------------------------------------------------

class TitansMAC(nn.Module):
    """
    Titans — Memory as a Context (MAC).

    Parameters
    ----------
    vocab_size   : int
    d_model      : int
    n_layers     : int   — number of MAC blocks
    mem_layers   : int   — depth of each NeuralMemory MLP (L_M)
    d_hidden     : int   — hidden dim of memory MLP
    n_persistent : int   — number of persistent-memory tokens
    chunk_size   : int   — segment / chunk length C
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
        chunk_size: int   = 128,
        ffn_mult: int     = 4,
        max_seq_len: int  = 8192,
        use_momentum: bool = True,
        use_decay: bool    = True,
    ):
        super().__init__()
        self.d_model      = d_model
        self.n_persistent = n_persistent

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            _MACBlock(
                d_model      = d_model,
                d_hidden     = d_hidden or (4 * d_model),
                mem_layers   = mem_layers,
                chunk_size   = chunk_size,
                n_persistent = n_persistent,
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
        x    = self.tok_emb(input_ids) + self.pos_emb(pos)   # (B, T, D)

        for block in self.blocks:
            x = block(x)

        x      = self.norm(x)
        logits = self.lm_head(x)

        out: dict = {"logits": logits}
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
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
