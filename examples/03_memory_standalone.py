"""
Example 3 — NeuralMemory standalone usage.
Shows retrieve-only inference and the surprise/momentum update in isolation.
"""

import torch
from titans.memory import NeuralMemory, PersistentMemory

D = 64
B = 2
T = 32

# -----------------------------------------------------------------------
# 1. NeuralMemory standalone
# -----------------------------------------------------------------------
mem = NeuralMemory(
    d_model    = D,
    n_layers   = 2,
    chunk_size = 8,
)

x   = torch.randn(B, T, D)
out = mem(x)                             # write + read
print(f"NeuralMemory write+read: {out.shape}")   # (B, T, D)

# Retrieval only (no weight update — eq. 15)
q      = torch.randn(B, 8, D)
h      = mem.retrieve(q)
print(f"NeuralMemory retrieve  : {h.shape}")     # (B, 8, D)

# -----------------------------------------------------------------------
# 2. PersistentMemory standalone
# -----------------------------------------------------------------------
pm    = PersistentMemory(n_tokens=16, d_model=D)
seq   = torch.randn(B, T, D)
aug   = pm(seq)                          # prepend P
print(f"PersistentMemory augmented seq: {aug.shape}")  # (B, 16+T, D)

out_stripped = pm.strip(aug)
print(f"PersistentMemory stripped     : {out_stripped.shape}")  # (B, T, D)

# Freeze at test time (task knowledge fixed)
pm.freeze()
print(f"P requires_grad after freeze: {pm.P.requires_grad}")

pm.unfreeze()
print(f"P requires_grad after unfreeze: {pm.P.requires_grad}")

# -----------------------------------------------------------------------
# 3. Memory depth ablation (L_M = 1, 2, 3, 4)
# -----------------------------------------------------------------------
print("\nMemory depth ablation (param counts):")
for lm in [1, 2, 3, 4]:
    m = NeuralMemory(d_model=D, n_layers=lm, chunk_size=8)
    n = sum(p.numel() for p in m.parameters())
    print(f"  L_M={lm}  →  {n:,} parameters")
