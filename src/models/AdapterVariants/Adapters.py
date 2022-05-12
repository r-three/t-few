import torch.nn as nn
from .VariantLayers import LowRankLinear, PHMLinear
from transformers.activations import ACT2FN


# From https://github.com/rabeehk/compacter

ACT2FN["identity"] = lambda x: x


class Adapter(nn.Module):
    def __init__(self, config, transformer_config):
        super().__init__()
        self.adapter_input_size = transformer_config.hidden_size
        self.adapter_latent_size = self.adapter_input_size // config.adapter_reduction_factor
        self.non_linearity = ACT2FN[config.adapter_non_linearity]
        self.residual = config.normal_adapter_residual

        # down projection
        self.down_proj = nn.Linear(self.adapter_input_size, self.adapter_latent_size)
        # up projection
        self.up_proj = nn.Linear(self.adapter_latent_size, self.adapter_input_size)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights -> so that initially we the whole Adapter layer is a near-identity function"""
        self.down_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.down_proj.bias.data.zero_()
        self.up_proj.weight.data.normal_(mean=0.0, std=0.02)
        self.up_proj.bias.data.zero_()

    def forward(self, x):
        output = self.up_proj(self.non_linearity(self.down_proj(x)))
        if self.residual:
            output = x + output
        return output


class LowRankAdapter(nn.Module):
    """This is the low-rank adapter, in which each adapter is composed of two rank-one matrices."""

    def __init__(self, config, transformer_config):
        super().__init__()
        self.config = config
        self.input_dim = transformer_config.hidden_size
        self.down_sample_size = self.input_dim // config.adapter_reduction_factor
        self.activation = ACT2FN[config.adapter_non_linearity]
        self.down_sampler = LowRankLinear(
            self.input_dim,
            self.down_sample_size,
            w_init=config.lowrank_adapter_w_init,
            rank=config.lowrank_adapter_rank,
        )
        self.up_sampler = LowRankLinear(
            self.down_sample_size,
            self.input_dim,
            w_init=config.lowrank_adapter_w_init,
            rank=config.lowrank_adapter_rank,
        )

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        output = self.up_sampler(z)
        return output


class HyperComplexAdapter(nn.Module):
    """Hypercomplex Adapter layer, in which the weights of up and down sampler modules
    are parameters are 1/n times of the conventional adapter layers, where n is
    hypercomplex division number."""

    def __init__(self, config, transformer_config):
        super().__init__()
        self.config = config
        self.input_dim = transformer_config.hidden_size
        self.down_sample_size = self.input_dim // config.adapter_reduction_factor
        self.activation = ACT2FN[config.adapter_non_linearity]
        self.down_sampler = PHMLinear(
            in_features=self.input_dim,
            out_features=self.down_sample_size,
            bias=True,
            c_init=config.compacter_phm_c_init,
            phm_dim=config.compacter_hypercomplex_division,
            learn_phm=config.compacter_learn_phm,
            w_init=config.compacter_hypercomplex_nonlinearity,
            shared_phm_rule=config.compacter_shared_phm_rule,
            factorized_phm=config.compacter_factorized_phm,
            shared_W_phm=config.compacter_shared_W_phm,
            factorized_phm_rule=config.compacter_factorized_phm_rule,
            phm_rank=config.compacter_phm_rank,
            phm_init_range=config.compacter_phm_init_range,
            kronecker_prod=config.compacter_kronecker_prod,
        )
        self.up_sampler = PHMLinear(
            in_features=self.down_sample_size,
            out_features=self.input_dim,
            bias=True,
            c_init=config.compacter_phm_c_init,
            phm_dim=config.compacter_hypercomplex_division,
            learn_phm=config.compacter_learn_phm,
            w_init=config.compacter_hypercomplex_nonlinearity,
            shared_phm_rule=config.compacter_shared_phm_rule,
            factorized_phm=config.compacter_factorized_phm,
            shared_W_phm=config.compacter_shared_W_phm,
            factorized_phm_rule=config.compacter_factorized_phm_rule,
            phm_rank=config.compacter_phm_rank,
            phm_init_range=config.compacter_phm_init_range,
            kronecker_prod=config.compacter_kronecker_prod,
        )

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        z = self.up_sampler(z)
        return x + z
