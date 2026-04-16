import torch
import pytest
from titans.utils.hf import TitansModelForCausalLM, TitansHFConfig

def test_hf_config_initialization():
    config = TitansHFConfig(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        mem_layers=1,
        variant='MAC'
    )
    assert config.model_type == 'titans'
    assert config.d_model == 128

def test_hf_model_forward():
    config = TitansHFConfig(
        vocab_size=1000,
        d_model=64,
        n_layers=1,
        mem_layers=1
    )
    model = TitansModelForCausalLM(config)
    input_ids = torch.randint(0, 1000, (1, 16))
    out = model(input_ids)
    assert 'logits' in out
    assert out['logits'].shape == (1, 16, 1000)

def test_hf_save_load(tmp_path):
    config = TitansHFConfig(vocab_size=100, d_model=32, n_layers=1, mem_layers=1)
    model = TitansModelForCausalLM(config)
    
    # Save state dict
    save_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), save_path)
    
    # Load
    new_model = TitansModelForCausalLM(config)
    new_model.load_state_dict(torch.load(save_path))
    assert new_model.model.d_model == 32
