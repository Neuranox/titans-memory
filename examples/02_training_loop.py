"""
Example 2 — Training loop on random data.
Demonstrates:
  • TitansConfig + build_model
  • build_optimizer + cosine LR schedule
  • Gradient clipping
  • CheckPointing
"""

import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from titans.utils import TitansConfig, build_model
from titans.utils.training import build_optimizer, get_cosine_schedule_with_warmup, count_parameters

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
cfg = TitansConfig(
    variant      = "MAG",
    vocab_size   = 512,
    d_model      = 128,
    n_layers     = 2,
    mem_layers   = 2,
    n_persistent = 4,
    chunk_size   = 16,
    window       = 32,
    ffn_mult     = 2,
    max_seq_len  = 64,
    lr           = 3e-4,
    weight_decay = 0.1,
    warmup_steps = 10,
    grad_clip    = 1.0,
    model_name   = "demo-mag",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# -----------------------------------------------------------------------
# Synthetic dataset: random token sequences
# -----------------------------------------------------------------------

N_SAMPLES  = 64
SEQ_LEN    = cfg.max_seq_len
data       = torch.randint(0, cfg.vocab_size, (N_SAMPLES, SEQ_LEN))
dataset    = TensorDataset(data)
loader     = DataLoader(dataset, batch_size=4, shuffle=True)

# -----------------------------------------------------------------------
# Model, optimizer, scheduler
# -----------------------------------------------------------------------
model  = build_model(cfg).to(device)
print(f"Parameters: {count_parameters(model):,}")

optim  = build_optimizer(model, lr=cfg.lr, weight_decay=cfg.weight_decay)
TOTAL  = len(loader) * 3                        # 3 epochs
sched  = get_cosine_schedule_with_warmup(optim, cfg.warmup_steps, TOTAL,
                                         min_lr_ratio=cfg.min_lr / cfg.lr)

# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------
model.train()
for epoch in range(3):
    epoch_loss = 0.0
    for step, (batch,) in enumerate(loader):
        ids    = batch.to(device)
        labels = ids.clone()

        optim.zero_grad()
        out  = model(ids, labels=labels)
        loss = out["loss"]
        loss.backward()

        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optim.step()
        sched.step()
        epoch_loss += loss.item()

    avg = epoch_loss / len(loader)
    lr  = sched.get_last_lr()[0]
    print(f"Epoch {epoch+1}/3  |  loss: {avg:.4f}  |  lr: {lr:.2e}")

# -----------------------------------------------------------------------
# Save checkpoint
# -----------------------------------------------------------------------
os.makedirs("checkpoints", exist_ok=True)
ckpt = {
    "config"     : cfg.to_dict(),
    "model_state": model.state_dict(),
    "optim_state": optim.state_dict(),
}
torch.save(ckpt, f"checkpoints/{cfg.model_name}.pt")
print("Checkpoint saved to checkpoints/demo-mag.pt")

# -----------------------------------------------------------------------
# Load checkpoint
# -----------------------------------------------------------------------
loaded = torch.load(f"checkpoints/{cfg.model_name}.pt", map_location=device)
cfg2   = TitansConfig(**loaded["config"])
model2 = build_model(cfg2).to(device)
model2.load_state_dict(loaded["model_state"])
model2.eval()
print("Checkpoint loaded successfully.")
