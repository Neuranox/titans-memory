"""
Example 1 — Quick-start: build any Titans variant and run a forward pass.
"""

import torch
from titans import TitansMAC, TitansMAG, TitansMAL, TitansLMM
from titans.utils import TitansConfig, build_model, count_parameters

# -----------------------------------------------------------------------
# 1. Build from config (recommended)
# -----------------------------------------------------------------------

cfg = TitansConfig.tiny(variant="MAC")        # ~30 M params, fast on CPU
cfg.vocab_size = 1_000                        # small vocab for demo
print("Config:", cfg)

model = build_model(cfg)
print(f"Parameters: {count_parameters(model):,}")

# -----------------------------------------------------------------------
# 2. Forward pass
# -----------------------------------------------------------------------

B, T = 2, 64                                  # batch=2, seq_len=64
input_ids = torch.randint(0, cfg.vocab_size, (B, T))
labels    = input_ids.clone()

out = model(input_ids, labels=labels)
print(f"Logits shape : {out['logits'].shape}")  # (2, 64, 1000)
print(f"Loss         : {out['loss'].item():.4f}")

# -----------------------------------------------------------------------
# 3. Text generation (greedy / top-k)
# -----------------------------------------------------------------------

prompt    = torch.randint(0, cfg.vocab_size, (1, 8))
generated = model.generate(prompt, max_new_tokens=20, top_k=40)
print(f"Generated ids: {generated[0].tolist()}")

# -----------------------------------------------------------------------
# 4. Instantiate each variant directly
# -----------------------------------------------------------------------

VOCAB = 1_000
D     = 128
NL    = 2

models = {
    "LMM": TitansLMM(VOCAB, d_model=D, n_layers=NL, mem_layers=2, n_persistent=4),
    "MAC": TitansMAC(VOCAB, d_model=D, n_layers=NL, mem_layers=2, n_persistent=4, chunk_size=16),
    "MAG": TitansMAG(VOCAB, d_model=D, n_layers=NL, mem_layers=2, n_persistent=4, window=32),
    "MAL": TitansMAL(VOCAB, d_model=D, n_layers=NL, mem_layers=2, n_persistent=4, window=32),
}

x = torch.randint(0, VOCAB, (1, 32))
for name, m in models.items():
    out = m(x)
    print(f"{name:4s} — logits: {out['logits'].shape}, params: {count_parameters(m):,}")
