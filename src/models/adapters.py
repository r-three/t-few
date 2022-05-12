import torch
import torch.nn as nn
import re
from .AdapterVariants.Adapters import Adapter, LowRankAdapter, HyperComplexAdapter


def get_adapter(adapter_type):
    if adapter_type == "normal":
        return Adapter
    elif adapter_type == "lowrank":
        return LowRankAdapter
    elif adapter_type == "compacter":
        return HyperComplexAdapter
    else:
        raise ValueError("Not Implemented")


class T5LayerFFWithAdapter(nn.Module):
    def __init__(self, T5LayerFF, config, transformer_config):
        super().__init__()
        self.DenseReluDense = T5LayerFF.DenseReluDense
        self.adapter = get_adapter(config.adapter_type)(config, transformer_config)
        self.layer_norm = T5LayerFF.layer_norm
        self.dropout = T5LayerFF.dropout

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        adapter_output = self.adapter(forwarded_states)
        hidden_states = hidden_states + self.dropout(adapter_output)
        return hidden_states


class T5LayerSelfAttentionWithAdapter(nn.Module):
    def __init__(self, T5LayerSelfAttention, config, transformer_config):
        super().__init__()
        self.SelfAttention = T5LayerSelfAttention.SelfAttention
        self.adapter = get_adapter(config.adapter_type)(config, transformer_config)
        self.layer_norm = T5LayerSelfAttention.layer_norm
        self.dropout = T5LayerSelfAttention.dropout

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        adapter_output = self.adapter(attention_output[0])
        hidden_states = hidden_states + self.dropout(adapter_output)
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttentionWithAdapter(nn.Module):
    def __init__(self, T5LayerCrossAttention, config, transformer_config):
        super().__init__()
        self.EncDecAttention = T5LayerCrossAttention.EncDecAttention
        self.adapter = get_adapter(config.adapter_type)(config, transformer_config)
        self.layer_norm = T5LayerCrossAttention.layer_norm
        self.dropout = T5LayerCrossAttention.dropout

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        adapter_output = self.adapter(attention_output[0])
        layer_output = hidden_states + self.dropout(adapter_output)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


def modify_with_adapters(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(".*block[.][0-9]*", m_name):
            layer = nn.ModuleList()
            if (config.compacter_add_compacter_in_self_attention):
                layer.append(
                    T5LayerSelfAttentionWithAdapter(
                        module.layer[0],
                        config,
                        transformer.config,
                    )
                )
            else:
                layer.append(module.layer[0])
            if module.is_decoder:
                if (config.compacter_add_compacter_in_cross_attention):
                    layer.append(
                        T5LayerCrossAttentionWithAdapter(
                            module.layer[1],
                            config,
                            transformer.config,
                        )
                    )
                else:
                    layer.append(module.layer[1])

            layer.append(
                T5LayerFFWithAdapter(
                    module.layer[2] if module.is_decoder else module.layer[1],
                    config,
                    transformer.config,
                )
            )
            module.layer = layer
    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--adapter_type",
        required=True,
        type=str,
        choices=["normal", "lowrank", "compacter"],
    )
    args = parser.parse_args()

    class AdapterConfig:
        def __init__(self, adapter_type):
            self.adapter_type = adapter_type

            if self.adapter_type == "normal":
                # Adapter Config
                self.adapter_reduction_factor = 16
                self.adapter_non_linearity = "relu"
                self.normal_adapter_residual = True
                self.add_compacter_in_attention = True
            elif self.adapter_type == "compacter":
                # Compacter
                self.adapter_reduction_factor = 16
                self.adapter_non_linearity = "relu"
                self.compacter_hypercomplex_division = 4
                self.compacter_learn_phm = True
                self.compacter_hypercomplex_nonlinearity = "xyz"
                self.compacter_shared_phm_rule = False
                self.compacter_factorized_phm = True
                self.compacter_shared_W_phm = False
                self.compacter_factorized_phm_rule = False
                self.compacter_phm_c_init = "xyz"
                self.compacter_phm_rank = 1
                self.compacter_phm_init_range = 0.0001
                self.compacter_kronecker_prod = False
                self.compacter_adapter_non_linearity = "gelu_new"
                self.compacter_add_compacter_in_attention = True

            self.trainable_param_names = ".*layer_norm.*|.*adapter.*"

    config = AdapterConfig(args.adapter_type)
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    input_seq = tokenizer(
        ["Applies a linear transformation to the incoming data."],
        return_tensors="pt",
    )
    target_seq = tokenizer(
        ["Parameters: in_features - size of each input sample. out_features - size of each output sample."],
        return_tensors="pt",
    )

    print("Old model")
    # print(model)
    # print(model.state_dict().keys())
    old_param = model.state_dict()
    with torch.no_grad():
        old_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    model = modify_with_adapters(model, config)
    new_param = model.state_dict()
    """
    for i in new_param.keys():
        if "adapter" in i:
            print(i, new_param[i])
    """
    # print(old_param - new_param)
    print("New model")
    # print(model)
    with torch.no_grad():
        new_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    print("Trainable parameters")
    """
    print(
        [
            p_name
            for p_name in dict(model.named_parameters()).keys()
            if re.fullmatch(config.trainable_param_names, p_name)
        ]
    )
    """
    print(f"Logits diff {torch.abs(old_outputs.logits - new_outputs.logits).mean():.3f}")
    print(f"Loss diff old={old_outputs.loss:.3f} new={new_outputs.loss:.3f}")
