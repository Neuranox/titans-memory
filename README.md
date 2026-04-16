# 🪐 Titans: Learning to Memorize at Test Time

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/pytorch-2.1%2B-orange)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An advanced, high-performance PyTorch implementation of the **Titans** architecture (Google Research, Jan 2025). This repo provides the tools to build models with **Infinite Context** using Neural Long-Term Memory.

---

## 📖 Table of Contents
1. [Core Concept](#core-concept)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Architecture Comparison](#architecture-comparison)
4. [Variants Detailed](#variants-detailed)
5. [Advanced Configuration](#advanced-configuration)
6. [Performance & Parallel Scan](#performance--parallel-scan)
7. [Project Structure](#project-structure)

---

## 🔍 Core Concept

Traditional Transformers use a fixed-size **Short-Term Memory** (Attention). As the sequence grows, the cost becomes quadratic ($O(T^2)$), and ancient information is eventually truncated. 

**Titans** solve this by adding a **Neural Memory branch**. This branch is a deep MLP that acts as an associative store. For every new token, the model:
1.  **Reads** from memory to get context.
2.  **Computes** the "surprise" (loss) of the new token.
3.  **Updates** its own weights via one step of gradient descent to "learn" the token.

---

## 🧮 Mathematical Foundation

The Neural Memory update follows these core equations from the paper:

### 1. Surprise Update (Momentum)
$$S_t = \eta_t S_{t-1} + \theta_t \nabla \ell(M_{t-1}; x_t)$$
Where $\nabla \ell$ is the gradient of the MSE loss between memory prediction and the actual value. $\eta$ is the surprise momentum.

### 2. Forgetting Gate (Weight Decay)
$$M_t = (1 - \alpha_t) M_{t-1} + S_t$$
The memory weights $M$ are updated using a combination of forgetting (weight decay) and the new surprise $S_t$.

---

## 📊 Architecture Comparison

| Feature | Transformers | RNN / LSTM | Mamba / SSM | **Titans (Ours)** |
|---|---|---|---|---|
| **Context Length** | Fixed (Linear/Quad) | Infinite (but lossy) | Infinite | **Infinite (High Fidelity)** |
| **Logic** | Matching | Compression | Linear Dynamics | **Test-Time Learning** |
| **Scaling** | $O(T^2)$ | $O(T)$ | $O(T)$ | **$O(T)$ (or $O(\log T)$)** |
| **Stability** | Very High | Low | High | **Very High** |

---

## 🧱 Variants Detailed

### **MAC (Memory as a Context)**
The gold standard for long-context RAG-style tasks.
- **Workflow:** `Retrieve Memory` -> `Prepend to Attention` -> `Full Attention`.
- **Best for:** Coding assistants, legal document analysis.

### **MAG (Memory as a Gate)**
- **Workflow:** Attention and Memory branches run in parallel; their outputs are gated via a SiLU-based mechanism.
- **Best for:** Creative writing and reasoning where short-term and long-term context must blend.

### **MAL (Memory as a Layer)**
- **Workflow:** A sequence is passed through Neural Memory, followed by a Sliding Window Attention layer.
- **Best for:** General-purpose LLMs seeking a balance between speed and precision.

---

## ⚙️ Advanced Configuration

Our `TitansConfig` allows for granular control over the memory dynamics:

```python
from titans.utils import TitansConfig

cfg = TitansConfig(
    variant="MAC",
    d_model=512,
    n_layers=12,
    mem_layers=2,        # Depth of the internal Neural Memory MLP
    n_persistent=16,     # Constant tokens that stay in memory
    chunk_size=64,       # Parallelization chunk size (Inner-loop)
    use_momentum=True,   # Enable η surprise flow
    use_decay=True       # Enable α forgetting gate
)
```

---

## ⚡ Performance & Parallel Scan

In version 0.3.0, we implemented a **Binary Tree Associative Scan**. 

**Why it matters:** Standard RNN-like updates must run token-by-token (one after another). Our associative scan allows the GPU to process entire chunks of a sequence at once by using the associative property of the linear recurrence, reducing latency from $O(T)$ to $O(\log T)$.

---

## 📂 Project Structure

```text
titans-memory/
├── titans/
│   ├── memory/           # Neural & Persistent Memory cores
│   ├── models/           # MAC, MAG, MAL, LMM variants
│   ├── ops/              # Parallel Associative Scan & Attention
│   └── utils/
│       ├── hf.py         # HuggingFace Transformers wrapper
│       ├── training.py   # DDP & Optimizer helpers
│       └── config.py     # Unified TitansConfig
├── tests/                # Full test suite (51+ tests)
├── scripts/              # Weight conversion & local scripts
├── examples/             # Quickstart & Training demos
├── pyproject.toml        # Build system & Dependencies
└── README.md
```

---
<p align="center">Developed with precision by the <b>Neuranox</b> team.</p>
