"""
Tests for titans.ops.scan  (associative scan).
"""
import pytest
import torch
from titans.ops.scan import parallel_scan, _scan_sequential


def _random_input(B, T, D, seed=42):
    torch.manual_seed(seed)
    eta = torch.sigmoid(torch.randn(B, T, D))    # in (0, 1)
    u   = torch.randn(B, T, D)
    return eta, u


class TestParallelScan:

    @pytest.mark.parametrize("T", [1, 2, 3, 7, 8, 15, 16, 17, 32, 64, 100])
    def test_matches_sequential(self, T):
        B, D   = 2, 8
        eta, u = _random_input(B, T, D)
        ref    = _scan_sequential(eta, u)
        out    = parallel_scan(eta, u)
        assert out.shape == (B, T, D), "Shape mismatch"
        assert torch.allclose(ref, out, atol=1e-5), \
            f"Max diff={(ref - out).abs().max():.2e} for T={T}"

    def test_single_token(self):
        eta, u = _random_input(1, 1, 4)
        out    = parallel_scan(eta, u)
        assert out.shape == (1, 1, 4)

    def test_output_shape_3d(self):
        B, T, D = 3, 20, 16
        eta, u  = _random_input(B, T, D)
        out     = parallel_scan(eta, u)
        assert out.shape == (B, T, D)

    def test_zero_decay(self):
        """eta=0 means S_t = u_t always (no memory)."""
        B, T, D = 1, 8, 4
        eta     = torch.zeros(B, T, D)
        u       = torch.randn(B, T, D)
        out     = parallel_scan(eta, u)
        assert torch.allclose(out, u, atol=1e-6)

    def test_unit_decay(self):
        """eta=1 means S_t = cumsum(u)."""
        B, T, D = 1, 8, 4
        eta     = torch.ones(B, T, D)
        u       = torch.randn(B, T, D)
        out     = parallel_scan(eta, u)
        ref     = _scan_sequential(eta, u)
        assert torch.allclose(out, ref, atol=1e-5)

    def test_backward_pass(self):
        """Gradients must flow through the scan."""
        B, T, D = 1, 8, 4
        eta_raw = torch.randn(B, T, D, requires_grad=True)
        u       = torch.randn(B, T, D, requires_grad=True)
        eta     = torch.sigmoid(eta_raw)
        out     = parallel_scan(eta, u)
        out.sum().backward()
        assert eta_raw.grad is not None
        assert u.grad is not None
