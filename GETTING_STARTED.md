# Getting Started with Titans-Memory (v0.2.0)

Congratulations on installing `titans-memory`! This package provides a high-performance, feature-rich implementation of the **Titans** (Learning to Memorize at Test Time) architecture.

## 🚀 Quick Start

```python
import torch
from titans.utils import TitansConfig, build_model

# Initialize a small MAC model (~30M parameters)
cfg = TitansConfig.tiny(variant='MAC', vocab_size=32000)
model = build_model(cfg)

# Forward pass
input_ids = torch.randint(0, 32000, (1, 512))
output = model(input_ids)
print(output['logits'].shape)  # [1, 512, 32000]
```

---

## ⚡ High-Performance Features (v0.2.0)

### 1. Parallel Associative Scan
Process massive sequences faster than standard implementations. Our `parallel_scan` uses a log-depth algorithm optimized for CUDA.

```python
from titans.ops import parallel_scan
# This is automatically used inside the model for GPU tensors
```

### 2. HuggingFace Integration
You can now use Titans models with the familiar `transformers` API:

```python
from titans.utils.hf import TitansModelForCausalLM, TitansHFConfig

config = TitansHFConfig(d_model=512, n_layers=12, variant='MAC')
model = TitansModelForCausalLM(config)

# Save/Load using standard HF methods
model.save_pretrained('./my-titans-model')
```

### 3. Distributed Training (DDP)
Scale your training across multiple GPUs seamlessly:

```python
from titans.utils.training import setup_ddp, wrap_ddp

local_rank = setup_ddp()
model = wrap_ddp(model, local_rank)
```

---

## 🛠 Model Variants

| Variant | Best For | Description |
|---|---|---|
| **MAC** | Long Context | Memory as Context (retrieves long-term memory as a prefix) |
| **MAG** | Gated Tasks | Memory as a Gate (Sliding Window Attention ⊗ NeuralMemory) |
| **MAL** | Sequential | Memory as a Layer |
| **LMM** | Efficiency | Standalone neural memory without attention |

---

## 📜 Citation

If you use this implementation in your research:

```bibtex
@article{behrouz2024titans,
  title   = {Titans: Learning to Memorize at Test Time},
  author  = {Ali Behrouz, Peilin Zhong, Vahab Mirrokni},
  year    = {2024}
}
```
