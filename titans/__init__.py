"""
Titans: Learning to Memorize at Test Time
==========================================
A PyTorch implementation of the Titans architecture from:
  "Titans: Learning to Memorize at Test Time"
  Ali Behrouz, Peilin Zhong, Vahab Mirrokni — Google Research (2024)
  arXiv:2501.00663

Public API
----------
from titans import (
    NeuralMemory,          # Long-term Memory Module (LMM)
    PersistentMemory,      # Task-knowledge memory
    TitansMAC,             # Memory as a Context
    TitansMAG,             # Memory as a Gate
    TitansMAL,             # Memory as a Layer
    TitansLMM,             # Standalone LMM (no attention)
)
"""

from titans.memory.neural_memory import NeuralMemory
from titans.memory.persistent_memory import PersistentMemory
from titans.models.mac import TitansMAC
from titans.models.mag import TitansMAG
from titans.models.mal import TitansMAL
from titans.models.lmm import TitansLMM

__version__ = "0.1.0"
__author__  = "Implementation based on Behrouz, Zhong & Mirrokni (2024)"

__all__ = [
    "NeuralMemory",
    "PersistentMemory",
    "TitansMAC",
    "TitansMAG",
    "TitansMAL",
    "TitansLMM",
]
