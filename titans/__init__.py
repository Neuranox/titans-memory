"""
Titans: Learning to Memorize at Test Time
==========================================
A PyTorch implementation of the Titans architecture from:
  "Titans: Learning to Memorize at Test Time"
  Ali Behrouz, Peilin Zhong, Vahab Mirrokni — Google Research (2024)
  arXiv:2501.00663

This library provides full implementations for the Neural Long-Term Memory (LMM) 
core module and the three major attention-hybrid architectures (MAC, MAG, MAL).

Core Architecture Variants
--------------------------
1. TitansMAC - Memory as a Context (Retrieving long-term memory as a context window prefix)
2. TitansMAG - Memory as a Gate (Parallel branch of Sliding Window Attention and LMM)
3. TitansMAL - Memory as a Layer (Sequential layers of LMM followed by Sliding Window Attention)
4. TitansLMM - Standalone Long-Term Memory Sequence Model (No Attention)

How to Use
----------
The easiest way to initialize a model is using `TitansConfig` and `build_model`.

1. Basic Inference:
    >>> import torch
    >>> from titans.utils import TitansConfig, build_model
    >>>
    >>> # Load the MAC variant matching the small paper scale (~170M parameters)
    >>> cfg = TitansConfig.small(variant="MAC")
    >>> cfg.vocab_size = 32000
    >>> model = build_model(cfg)
    >>> 
    >>> # Forward pass
    >>> input_ids = torch.randint(0, 32000, (1, 512))
    >>> out = model(input_ids)
    >>> logits = out["logits"]

2. Text Generation:
    All model variants come with a `.generate()` utility for autoregressive decoding:
    >>> prompt = torch.randint(0, 32000, (1, 10))
    >>> generated_ids = model.generate(prompt, max_new_tokens=50, top_k=40)

3. Custom Training:
    If developing your own training loop, `forward` automatically calculates Cross Entropy 
    loss if `labels` are provided. The package includes cosine-warmup schedulers mirroring the paper.
    >>> from titans.utils.training import build_optimizer, get_cosine_schedule_with_warmup
    >>> 
    >>> optim = build_optimizer(model, lr=4e-4, weight_decay=0.1)
    >>> sched = get_cosine_schedule_with_warmup(optim, warmup_steps=2000, total_steps=100000)
    >>> 
    >>> out = model(input_ids, labels=input_ids)
    >>> loss = out["loss"]
    >>> loss.backward()
    >>> optim.step()
    
4. Modular Usage (Neural Memory Only):
    You can extract and use the deep `NeuralMemory` MLP and the `PersistentMemory` 
    modules natively within any custom PyTorch code without using the provided Encoders!
    >>> from titans import NeuralMemory
    >>> memory = NeuralMemory(d_model=512, d_hidden=2048, n_layers=2)
    >>> token_embeddings = torch.randn(2, 64, 512)
    >>> processed_state = memory(token_embeddings)
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
