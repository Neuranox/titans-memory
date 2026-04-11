"""
Tests for all Titans model variants (forward pass, loss, generation).
"""
import pytest
import torch
from titans import TitansLMM, TitansMAC, TitansMAG, TitansMAL
from titans.utils import TitansConfig, build_model, count_parameters

VOCAB = 64
D     = 32
NL    = 2
NP    = 4
C     = 8
W     = 16
B     = 2
T     = 24


def _ids(b=B, t=T, v=VOCAB):
    return torch.randint(0, v, (b, t))


# ---------------------------------------------------------------------------
# LMM
# ---------------------------------------------------------------------------
class TestTitansLMM:

    @pytest.fixture
    def model(self):
        return TitansLMM(VOCAB, d_model=D, n_layers=NL, mem_layers=2,
                         n_persistent=NP, chunk_size=C)

    def test_logits_shape(self, model):
        out = model(_ids())
        assert out["logits"].shape == (B, T, VOCAB)

    def test_loss(self, model):
        ids = _ids()
        out = model(ids, labels=ids)
        assert "loss" in out
        assert out["loss"].item() > 0

    def test_generate(self, model):
        prompt = _ids(b=1, t=4)
        gen    = model.generate(prompt, max_new_tokens=5, top_k=10)
        assert gen.shape == (1, 9)


# ---------------------------------------------------------------------------
# MAC
# ---------------------------------------------------------------------------
class TestTitansMAC:

    @pytest.fixture
    def model(self):
        return TitansMAC(VOCAB, d_model=D, n_layers=NL, mem_layers=2,
                         n_persistent=NP, chunk_size=C)

    def test_logits_shape(self, model):
        out = model(_ids())
        assert out["logits"].shape == (B, T, VOCAB)

    def test_loss(self, model):
        ids = _ids()
        out = model(ids, labels=ids)
        assert out["loss"].item() > 0

    def test_generate(self, model):
        prompt = _ids(b=1, t=4)
        gen    = model.generate(prompt, max_new_tokens=5)
        assert gen.shape == (1, 9)


# ---------------------------------------------------------------------------
# MAG
# ---------------------------------------------------------------------------
class TestTitansMAG:

    @pytest.fixture
    def model(self):
        return TitansMAG(VOCAB, d_model=D, n_layers=NL, mem_layers=2,
                         n_persistent=NP, chunk_size=C, window=W)

    def test_logits_shape(self, model):
        out = model(_ids())
        assert out["logits"].shape == (B, T, VOCAB)

    def test_loss(self, model):
        ids = _ids()
        out = model(ids, labels=ids)
        assert out["loss"].item() > 0


# ---------------------------------------------------------------------------
# MAL
# ---------------------------------------------------------------------------
class TestTitansMAL:

    @pytest.fixture
    def model(self):
        return TitansMAL(VOCAB, d_model=D, n_layers=NL, mem_layers=2,
                         n_persistent=NP, chunk_size=C, window=W)

    def test_logits_shape(self, model):
        out = model(_ids())
        assert out["logits"].shape == (B, T, VOCAB)

    def test_loss(self, model):
        ids = _ids()
        out = model(ids, labels=ids)
        assert out["loss"].item() > 0


# ---------------------------------------------------------------------------
# Config + Factory
# ---------------------------------------------------------------------------
class TestConfigAndFactory:

    def test_tiny_config(self):
        cfg   = TitansConfig.tiny(variant="MAG")
        cfg.vocab_size = VOCAB
        model = build_model(cfg)
        out   = model(_ids())
        assert out["logits"].shape[-1] == VOCAB

    def test_config_json_roundtrip(self, tmp_path):
        cfg  = TitansConfig.small(variant="MAL")
        path = str(tmp_path / "cfg.json")
        cfg.to_json(path)
        cfg2 = TitansConfig.from_json(path)
        assert cfg == cfg2

    def test_count_parameters(self):
        m = TitansLMM(VOCAB, d_model=D, n_layers=1, mem_layers=1,
                      n_persistent=0, chunk_size=4)
        n = count_parameters(m)
        assert n > 0

    @pytest.mark.parametrize("variant", ["MAC", "MAG", "MAL", "LMM"])
    def test_all_variants_build(self, variant):
        cfg = TitansConfig(variant=variant, vocab_size=VOCAB, d_model=D,
                           n_layers=NL, mem_layers=1, n_persistent=4,
                           chunk_size=C, window=W, max_seq_len=T)
        model = build_model(cfg)
        out   = model(_ids())
        assert out["logits"].shape == (B, T, VOCAB)

    def test_invalid_variant_raises(self):
        cfg = TitansConfig(variant="INVALID")
        with pytest.raises(ValueError):
            build_model(cfg)
