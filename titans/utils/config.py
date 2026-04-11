"""
TitansConfig — single dataclass holding all hyperparameters.
Supports serialisation to / from JSON for reproducibility.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional


@dataclass
class TitansConfig:
    """
    Unified configuration for all Titans variants.

    Architecture
    ------------
    variant      : 'MAC' | 'MAG' | 'MAL' | 'LMM'
    vocab_size   : vocabulary size
    d_model      : hidden dimension
    n_layers     : number of transformer / memory blocks
    mem_layers   : depth of each NeuralMemory MLP  (L_M ≥ 1)
    d_hidden     : hidden dim of memory MLP (None → 4*d_model)
    n_persistent : number of persistent-memory tokens (0 = disabled)
    ffn_mult     : FFN hidden-dim multiplier

    Memory
    ------
    chunk_size   : tokens per mini-batch / segment (inner-loop)
    use_momentum : enable surprise-momentum update (η)
    use_decay    : enable forgetting gate / weight-decay (α)

    Attention (MAC / MAG / MAL only)
    ---------------------------------
    window       : sliding-window size W for SWA (MAG / MAL)

    Training
    --------
    max_seq_len  : maximum sequence length
    lr           : peak learning rate
    min_lr       : minimum learning rate (cosine schedule)
    weight_decay : AdamW weight decay
    beta1, beta2 : AdamW betas
    warmup_steps : linear warm-up steps
    grad_clip    : gradient clipping max norm (0 = disabled)
    batch_size   : tokens per global batch
    train_tokens : total training tokens

    I/O
    ---
    model_name   : string identifier (used for checkpoints)
    """

    # ---- Architecture ----
    variant: Literal["MAC", "MAG", "MAL", "LMM"] = "MAC"
    vocab_size:   int = 32_000
    d_model:      int = 512
    n_layers:     int = 12
    mem_layers:   int = 2
    d_hidden:     Optional[int] = None        # None → 4 * d_model
    n_persistent: int = 16
    ffn_mult:     int = 4

    # ---- Memory ----
    chunk_size:   int  = 64
    use_momentum: bool = True
    use_decay:    bool = True

    # ---- Attention ----
    window: int = 512

    # ---- Training ----
    max_seq_len:  int   = 4_096
    lr:           float = 4e-4
    min_lr:       float = 4e-5
    weight_decay: float = 0.1
    beta1:        float = 0.9
    beta2:        float = 0.95
    warmup_steps: int   = 2_000
    grad_clip:    float = 1.0
    batch_size:   int   = 524_288          # 0.5 M tokens (paper default)
    train_tokens: int   = 15_000_000_000  # 15 B tokens

    # ---- I/O ----
    model_name: str = "titans"

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_json(self, path: str) -> None:
        """Save config to a JSON file."""
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TitansConfig":
        """Load config from a JSON file."""
        with open(path) as f:
            d = json.load(f)
        return cls(**d)

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------------
    # Pre-built configs matching paper scales
    # ------------------------------------------------------------------

    @classmethod
    def tiny(cls, variant: str = "MAC") -> "TitansConfig":
        """~30 M parameter model for quick experiments."""
        return cls(variant=variant, d_model=256, n_layers=6, mem_layers=2,
                   n_persistent=8, chunk_size=32, model_name=f"titans-tiny-{variant.lower()}")

    @classmethod
    def small(cls, variant: str = "MAC") -> "TitansConfig":
        """~170 M parameter model."""
        return cls(variant=variant, d_model=512, n_layers=12, mem_layers=2,
                   n_persistent=16, chunk_size=64, model_name=f"titans-170m-{variant.lower()}")

    @classmethod
    def medium(cls, variant: str = "MAC") -> "TitansConfig":
        """~340 M parameter model (Table 1 baseline)."""
        return cls(variant=variant, d_model=768, n_layers=24, mem_layers=2,
                   n_persistent=16, chunk_size=64, model_name=f"titans-340m-{variant.lower()}")

    @classmethod
    def large(cls, variant: str = "MAC") -> "TitansConfig":
        """~760 M parameter model (Table 1 largest)."""
        return cls(variant=variant, d_model=1024, n_layers=32, mem_layers=4,
                   n_persistent=32, chunk_size=128, weight_decay=0.1,
                   train_tokens=30_000_000_000,
                   model_name=f"titans-760m-{variant.lower()}")
