"""
Neural Long-Term Memory Module (LMM)
=====================================
Core contribution from §3 of "Titans: Learning to Memorize at Test Time".

Architecture
------------
The memory M is a deep MLP with L_M layers.  For each incoming token x_t:

  1. **Write** (inner-loop gradient descent with momentum + weight decay):
        k_t = x_t @ W_K               (key projection)
        v_t = x_t @ W_V               (value projection)
        loss = ||M(k_t) - v_t||²
        S_t  = eta_t * S_{t-1}  -  theta_t * grad_loss   (surprise w/ momentum)
        M_t  = (1 - alpha_t) * M_{t-1}  +  S_t           (weight decay forgetting)

  2. **Read** (forward pass, no weight update):
        q_t = x_t @ W_Q
        y_t = M*(q_t)                  (M* = inference only)

Parallelisation
---------------
Inside each chunk of size `chunk_size`:
  - All gradients ∇ℓ(M_0; x_i) are computed in a *batched* manner using
    the closed-form gradient of the MSE loss (linear-memory approximation
    used for the batch-gradient computation, following Sun et al. 2024).
  - Momentum states S_t are then computed via the parallel associative scan
    (`titans.ops.scan.parallel_scan`).

Between chunks the memory state is updated sequentially (one chunk at a time).

For LM ≥ 2 MLP layers the *gradient of the final layer only* is used to
steer the update (gradient w.r.t. the output projection), making the inner
loop efficient while retaining non-linearity in the memory network.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from titans.ops.scan import parallel_scan


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class _MemoryMLP(nn.Module):
    """
    Simple L_M-layer MLP used as the neural memory.
    Hidden dim = dim_hidden, activation = SiLU (as in the paper §4.4).
    """

    def __init__(self, d_in: int, d_hidden: int, n_layers: int = 2):
        super().__init__()
        assert n_layers >= 1, "Memory must have at least 1 layer."
        layers: list[nn.Module] = []
        in_dim = d_in
        for i in range(n_layers):
            out_dim = d_in if i == n_layers - 1 else d_hidden
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            if i < n_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:          # (*, d_in) -> (*, d_in)
        return self.net(x)


# ---------------------------------------------------------------------------
# NeuralMemory – public class
# ---------------------------------------------------------------------------

class NeuralMemory(nn.Module):
    """
    Neural Long-Term Memory Module (LMM).

    Parameters
    ----------
    d_model      : int   — token embedding dimension
    d_hidden     : int   — hidden size of the memory MLP (default = 4 * d_model)
    n_layers     : int   — depth of the memory MLP (L_M ≥ 1; paper uses 2–4)
    chunk_size   : int   — tokens per mini-batch for the parallelised inner loop
    init_lr      : float — initial (learnable) inner-loop learning rate θ
    use_momentum : bool  — enable surprise momentum (η)
    use_decay    : bool  — enable weight-decay / forgetting (α)

    Shape convention
    ----------------
    All tensors: (Batch, SeqLen, d_model)
    """

    def __init__(
        self,
        d_model: int,
        d_hidden: Optional[int] = None,
        n_layers: int = 2,
        chunk_size: int = 64,
        init_lr: float = 1e-3,
        use_momentum: bool = True,
        use_decay: bool = True,
    ):
        super().__init__()
        self.d_model    = d_model
        self.d_hidden   = d_hidden or (4 * d_model)
        self.n_layers   = n_layers
        self.chunk_size = chunk_size
        self.use_momentum = use_momentum
        self.use_decay    = use_decay

        # ---- Memory network (the "fast weights" / associative store) ----
        self.memory = _MemoryMLP(d_model, self.d_hidden, n_layers)

        # ---- Projections for key / value / query (eq. 11) ----
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.W_V = nn.Linear(d_model, d_model, bias=False)
        self.W_Q = nn.Linear(d_model, d_model, bias=False)

        # ---- 1-D depthwise separable conv (§4.4) ----
        self.conv_k = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1,
                                groups=d_model, bias=False)
        self.conv_v = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1,
                                groups=d_model, bias=False)
        self.conv_q = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1,
                                groups=d_model, bias=False)

        # ---- Data-dependent gating parameters (per token, projected from x) ----
        #      theta_t  – learning-rate gate       (momentary surprise weight)
        #      eta_t    – momentum decay gate      (past surprise retention)
        #      alpha_t  – forgetting gate          (weight decay strength)
        self.gate_theta = nn.Linear(d_model, d_model, bias=True)
        self.gate_eta   = nn.Linear(d_model, d_model, bias=True)
        self.gate_alpha = nn.Linear(d_model, d_model, bias=True)

        # ---- Output gating + projection (§4.4) ----
        self.out_norm  = nn.LayerNorm(d_model)
        self.out_gate  = nn.Linear(d_model, d_model, bias=False)
        self.out_proj  = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # ---- Gating helpers ----
    # ------------------------------------------------------------------

    def _theta(self, x: Tensor) -> Tensor:
        """θ_t: how much of momentary surprise to incorporate."""
        return torch.sigmoid(self.gate_theta(x))           # (B, T, D)

    def _eta(self, x: Tensor) -> Tensor:
        """η_t: how much of past surprise to retain (momentum decay)."""
        return torch.sigmoid(self.gate_eta(x))             # (B, T, D)

    def _alpha(self, x: Tensor) -> Tensor:
        """α_t: how much old memory to forget (weight decay)."""
        return torch.sigmoid(self.gate_alpha(x))           # (B, T, D)

    # ------------------------------------------------------------------
    # ---- Core forward ----
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Process sequence x through the neural memory.

        Parameters
        ----------
        x : (B, T, d_model)

        Returns
        -------
        out : (B, T, d_model)   — retrieved memory for each position
        """
        B, T, D = x.shape

        # ---- Project + conv ----
        # transpose for Conv1d: (B, D, T)
        k = self.conv_k(self.W_K(x).transpose(1, 2)).transpose(1, 2)  # (B,T,D)
        v = self.conv_v(self.W_V(x).transpose(1, 2)).transpose(1, 2)
        q = self.conv_q(self.W_Q(x).transpose(1, 2)).transpose(1, 2)

        # ℓ2-normalise keys and queries (§4.4)
        k = F.normalize(k, dim=-1)
        q = F.normalize(q, dim=-1)

        # ---- Data-dependent gates ----
        theta = self._theta(x)    # (B, T, D)
        eta   = self._eta(x)      # (B, T, D)
        alpha = self._alpha(x)    # (B, T, D)

        # ---- Retrieve memories chunk-by-chunk ----
        retrieved = self._retrieve_chunked(q, k, v, theta, eta, alpha)   # (B, T, D)

        # ---- Output gating (§4.4) ----
        out = self.out_norm(retrieved)
        out = out * torch.sigmoid(self.out_gate(x))
        out = self.out_proj(out)
        return out

    # ------------------------------------------------------------------
    # ---- Chunked write + read ----
    # ------------------------------------------------------------------

    def _get_h(self, x: Tensor) -> Tensor:
        """Get the hidden representation just before the last layer."""
        for module in list(self.memory.net)[:-1]:
            x = module(x)
        return x

    def _retrieve_chunked(
        self,
        q: Tensor, k: Tensor, v: Tensor,
        theta: Tensor, eta: Tensor, alpha: Tensor,
    ) -> Tensor:
        """
        Walk through the sequence in chunks of size `chunk_size`.

        Inside each chunk we:
          1. Compute batched gradients (closed-form MSE grad for the linear
             approximation: ∇ = (M(k) - v) * ... ).
          2. Run parallel_scan to get momentum states S_t.
          3. Update the memory weights (accumulated additive update to W_out).
          4. Read (forward pass) to get retrieved values.

        Between chunks the memory state carries over (sequential across chunks).
        """
        B, T, D = q.shape
        C = self.chunk_size

        all_retrieved: list[Tensor] = []

        # We keep a *delta* on the output projection of the last memory layer
        # as the "fast weight" that accumulates across tokens.
        # Shape: (B, D_out, D_in) where D_in is the last linear's in_features.
        last_linear: nn.Linear = self.memory.net[-1]          # last Linear layer
        W0 = last_linear.weight.detach().clone()               # (D_out, D_in)
        # Per-sample fast-weight delta, shape (B, D_out, D_in)
        fast_W = W0.unsqueeze(0).expand(B, -1, -1).clone()

        # Momentum state S: (B, D_out, D_in)
        D_out = last_linear.out_features
        D_in  = last_linear.in_features
        S = torch.zeros(B, D_out, D_in, dtype=q.dtype, device=q.device)

        # Precompute the input to the last layer for both k and q
        h_k = self._get_h(k)
        h_q = self._get_h(q)

        for start in range(0, T, C):
            end = min(start + C, T)
            c   = end - start          # actual chunk length

            h_q_c   = h_q  [:, start:end, :]   # (B, c, D_in)
            h_k_c   = h_k  [:, start:end, :]
            v_c     = v    [:, start:end, :]
            theta_c = theta[:, start:end, :]   # (B, c, D)
            eta_c   = eta  [:, start:end, :]
            alpha_c = alpha[:, start:end, :]

            # ---- Compute gradients (batched, eq. 17) ----
            # For the linear MLP last-layer approximation:
            #   grad_W = (W * h_k - v)^T @ h_k
            # We compute using the current fast_W per sample.
            # h_k_c: (B, c, D_in)  v_c: (B, c, D_out)

            # forward prediction using fast weights: (B, c, D_out)
            pred = torch.bmm(h_k_c, fast_W.transpose(1, 2))   # (B, c, D_out)
            # residual: (B, c, D_out)
            residual = pred - v_c

            # grad wrt W per token: outer product residual ⊗ h_k
            # grad[b, t] shape: (D_out, D_in)
            # Batched: (B, c, D_out, D_in)
            grad_per_token = torch.einsum("bto,bti->btoi", residual, h_k_c)

            # Flatten last two dims for the scan: (B, c, D_out*D_in)
            flat_grad  = grad_per_token.reshape(B, c, D_out * D_in)
            flat_eta   = eta_c.mean(-1, keepdim=True).expand(B, c, D_out * D_in)
            flat_theta = theta_c.mean(-1, keepdim=True).expand(B, c, D_out * D_in)

            # u_t = -theta_t * grad_t   (innovation for the scan)
            u = -flat_theta * flat_grad            # (B, c, D_out*D_in)

            # ---- Parallel scan to get momentum states  S_t = eta_t*S_{t-1} + u_t
            if self.use_momentum:
                S_flat_init = S.reshape(B, 1, D_out * D_in)
                # prepend initial state as "offset" for the very first element
                # (We roll the scan: the first input feeds S_{-1}=S_prev)
                # Simple approach: inject S into u[0] by running sequential for
                # the first element and parallel scan for the rest.
                u_prime = u.clone()
                u_prime[:, 0, :] = flat_eta[:, 0, :] * S_flat_init.squeeze(1) + u[:, 0, :]
                S_chunk = parallel_scan(flat_eta, u_prime)     # (B, c, D_out*D_in)
            else:
                S_chunk = u   # no momentum – use momentary surprise only

            # Last S of this chunk becomes S for next chunk
            S = S_chunk[:, -1, :].reshape(B, D_out, D_in)

            # ---- Apply forgetting + accumulate weight update ----
            # alpha averaged over hidden dim for weight-decay scalar per token
            alpha_scalar = alpha_c.mean(-1)           # (B, c)

            # Fold S_chunk back to weight-update shape
            delta_W_chunk = S_chunk.reshape(B, c, D_out, D_in)   # per-token delta

            # Apply token-by-token with forgetting gate (eq. 13)
            # M_t = (1 - alpha_t) * M_{t-1} + S_t
            if self.use_decay:
                for t_local in range(c):
                    a = alpha_scalar[:, t_local].unsqueeze(-1).unsqueeze(-1)   # (B,1,1)
                    fast_W = (1.0 - a) * fast_W + delta_W_chunk[:, t_local, :, :]
            else:
                # No decay: just accumulate the last delta
                fast_W = fast_W + delta_W_chunk[:, -1, :, :]

            # ---- Read: forward pass with updated fast_W ----
            # h_q_c: (B, c, D_in)  →  out: (B, c, D_out)
            retrieved_chunk = torch.bmm(h_q_c, fast_W.transpose(1, 2))
            all_retrieved.append(retrieved_chunk)

        return torch.cat(all_retrieved, dim=1)    # (B, T, D)

    # ------------------------------------------------------------------
    # ---- Inference-only retrieval (no weight update) ----
    # ------------------------------------------------------------------

    def retrieve(self, q: Tensor) -> Tensor:
        """
        Retrieve from memory without updating weights (eq. 15: y = M*(q)).
        Used in the MAC architecture to read historical context.

        q   : (B, T, d_model)
        out : (B, T, d_model)
        """
        q_proj = self.conv_q(self.W_Q(q).transpose(1, 2)).transpose(1, 2)
        q_proj = F.normalize(q_proj, dim=-1)
        return self.memory(q_proj)
