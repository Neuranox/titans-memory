import torch
from transformers import PreTrainedModel, PretrainedConfig
from titans import TitansMAC, TitansMAG, TitansMAL, TitansLMM
from titans.utils import TitansConfig, build_model

class TitansHFConfig(PretrainedConfig):
    model_type = "titans"
    def __init__(self, tie_word_embeddings=True, **kwargs):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.tie_word_embeddings = tie_word_embeddings
        for key, value in kwargs.items():
            setattr(self, key, value)

class TitansModelForCausalLM(PreTrainedModel):
    config_class = TitansHFConfig
    def __init__(self, config):
        super().__init__(config)
        t_cfg = TitansConfig(
            variant=getattr(config, "variant", "MAC"),
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_layers=config.n_layers,
            mem_layers=config.mem_layers,
        )
        self.model = build_model(t_cfg)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.tok_emb

    def set_input_embeddings(self, value):
        self.model.tok_emb = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self.model.lm_head.weight = self.model.tok_emb.weight
            
    def tie_weights(self, *args, **kwargs):
        self._tie_weights()
        super().tie_weights(*args, **kwargs)

    def forward(self, input_ids, labels=None, **kwargs):
        return self.model(input_ids, labels=labels)

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)
