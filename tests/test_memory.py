"""
Tests for NeuralMemory and PersistentMemory modules.
"""
import pytest
import torch
from titans.memory import NeuralMemory, PersistentMemory


class TestNeuralMemory:

    @pytest.fixture
    def mem(self):
        return NeuralMemory(d_model=32, n_layers=2, chunk_size=8)

    def test_forward_shape(self, mem):
        x   = torch.randn(2, 16, 32)
        out = mem(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_retrieve_shape(self, mem):
        q   = torch.randn(2, 8, 32)
        out = mem.retrieve(q)
        assert out.shape == q.shape

    @pytest.mark.parametrize("lm", [1, 2, 3])
    def test_memory_depth(self, lm):
        m   = NeuralMemory(d_model=16, n_layers=lm, chunk_size=4)
        x   = torch.randn(1, 8, 16)
        out = m(x)
        assert out.shape == x.shape

    def test_no_momentum(self):
        m   = NeuralMemory(d_model=16, n_layers=1, chunk_size=4, use_momentum=False)
        x   = torch.randn(1, 8, 16)
        out = m(x)
        assert out.shape == x.shape

    def test_no_decay(self):
        m   = NeuralMemory(d_model=16, n_layers=1, chunk_size=4, use_decay=False)
        x   = torch.randn(1, 8, 16)
        out = m(x)
        assert out.shape == x.shape

    def test_gradients_flow(self, mem):
        x   = torch.randn(1, 8, 32, requires_grad=True)
        out = mem(x)
        out.sum().backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_chunk_size_larger_than_seq(self):
        m   = NeuralMemory(d_model=16, n_layers=1, chunk_size=128)
        x   = torch.randn(1, 10, 16)          # T < chunk_size
        out = m(x)
        assert out.shape == x.shape


class TestPersistentMemory:

    def test_prepend_shape(self):
        pm  = PersistentMemory(n_tokens=8, d_model=32)
        x   = torch.randn(2, 16, 32)
        out = pm(x)
        assert out.shape == (2, 24, 32)

    def test_strip_shape(self):
        pm  = PersistentMemory(n_tokens=8, d_model=32)
        x   = torch.randn(2, 16, 32)
        aug = pm(x)
        stripped = pm.strip(aug)
        assert stripped.shape == x.shape

    def test_freeze_unfreeze(self):
        pm = PersistentMemory(n_tokens=4, d_model=8)
        pm.freeze()
        assert not pm.P.requires_grad
        pm.unfreeze()
        assert pm.P.requires_grad

    def test_gradient_through_persistent(self):
        pm  = PersistentMemory(n_tokens=4, d_model=8)
        x   = torch.randn(1, 4, 8)
        aug = pm(x)
        aug.sum().backward()
        assert pm.P.grad is not None
