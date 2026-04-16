import torch
import time
import pytest
from titans.ops.scan import _scan_sequential, _scan_parallel_log

@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA not available')
def test_scan_speedup():
    B, T, D = 4, 1024, 512
    eta = torch.rand(B, T, D).cuda()
    u = torch.randn(B, T, D).cuda()
    
    # Warmup
    _ = _scan_sequential(eta, u)
    _ = _scan_parallel_log(eta, u)
    
    # Time Sequential
    start = time.perf_counter()
    for _ in range(5): _ = _scan_sequential(eta, u)
    t_seq = time.perf_counter() - start
    
    # Time Parallel
    start = time.perf_counter()
    for _ in range(5): _ = _scan_parallel_log(eta, u)
    t_par = time.perf_counter() - start
    
    print(f'\nSequential: {t_seq:.4f}s | Parallel: {t_par:.4f}s')
    print(f'Speed-up: {t_seq/t_par:.2f}x')
    
    assert t_par < t_seq, 'Parallel scan should be faster on GPU'
