import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer
from src.utils.get_optimizer import get_optimizer


class T5AttentionPrefixTuning(nn.Module):
    def __init__(self, attention_layer, num_prefix_tokens, parameterization, shared=None):
        super().__init__()
        self.is_decoder = attention_layer.is_decoder
        self.has_relative_attention_bias = attention_layer.has_relative_attention_bias

        self.relative_attention_num_buckets = attention_layer.relative_attention_num_buckets
        self.d_model = attention_layer.d_model
        self.key_value_proj_dim = attention_layer.key_value_proj_dim
        self.n_heads = attention_layer.n_heads
        self.dropout = attention_layer.dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        self.prune_heads = attention_layer.prune_heads
        self._relative_position_bucket = attention_layer._relative_position_bucket
        self.compute_bias = attention_layer.compute_bias

        self.q = attention_layer.q
        self.k = attention_layer.k
        self.v = attention_layer.v
        self.o = attention_layer.o
        if self.has_relative_attention_bias:
            self.relative_attention_bias = attention_layer.relative_attention_bias
        self.pruned_heads = attention_layer.pruned_heads
        self.gradient_checkpointing = attention_layer.gradient_checkpointing

        self.parameterization = parameterization
        self.num_prefix_tokens = num_prefix_tokens
        self.mode = "apply"

        self.setup_prefix(shared)

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Modified from T5Attention forward
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        pask_key_value, query_length, use_cache disabled
        """
        assert past_key_value is None
        assert query_length is None
        assert not use_cache
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
        # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
        batch_size, seq_length = hidden_states.shape[:2]
        key_length = seq_length if key_value_states is None else key_value_states.shape[1]

        def shape(states):
            """projection"""
            return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

        def unshape(states):
            """reshape"""
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

        def project(hidden_states, proj_layer, key_value_states, prefix_states):
            """projects hidden states correctly to key/query states"""
            if key_value_states is None:
                # self-attn
                output_states = proj_layer(hidden_states)
            else:
                # cross-attn
                output_states = proj_layer(key_value_states)
            if prefix_states is not None:
                output_states = torch.cat([prefix_states, output_states], dim=1)
            return output_states

        # get query states
        query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

        # get key/value states
        if self.mode == "apply":
            prefix = self.get_prefix(batch_size)
            key_length += self.num_prefix_tokens
        else:
            prefix = (None, None)

        key_states = project(hidden_states, self.k, key_value_states, prefix[0])
        value_states = project(hidden_states, self.v, key_value_states, prefix[1])

        if self.mode == "store":
            self.stored_key_value_states = (key_states, value_states)

        key_states, value_states = shape(key_states), shape(value_states)

        # compute scores
        scores = torch.matmul(
            query_states, key_states.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, key_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                if self.mode == "apply":
                    position_bias = self.compute_bias(seq_length + self.num_prefix_tokens, key_length)[
                        :, :, -seq_length:, :
                    ]
                else:
                    position_bias = self.compute_bias(seq_length, key_length)

            if mask is not None:
                if self.mode == "apply":
                    mask = F.pad(mask, value=-0.0, pad=(self.num_prefix_tokens, 0))
                position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

        scores += position_bias
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
            scores
        )  # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.dropout(
            attn_weights, p=self.dropout, training=self.training
        )  # (batch_size, n_heads, seq_length, key_length)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
        attn_output = self.o(attn_output)

        outputs = (attn_output,) + (None,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs

    def setup_prefix(self, shared):
        if self.parameterization.startswith("mlp"):
            # hidden_size = int(self.parameterization.split("-")[1])
            # nn.Embedding(num_prefix_tokens, transformer_config.d_model)
            # nn.Linear(transformer_config.d_model, hidden_size),
            self.prefix_emb = shared["prefix_emb"]
            self.prefix_mlp = nn.Sequential(
                shared["prefix_linear"],
                nn.Tanh(),
                nn.Linear(shared["prefix_linear"].out_features, self.inner_dim * 2),
            )
        elif self.parameterization == "direct":
            self.prefix_direct = nn.Parameter(torch.randn(self.num_prefix_tokens, self.inner_dim * 2))
        else:
            raise NotImplementedError()

    def get_prefix(self, bs):
        if self.parameterization.startswith("mlp"):
            prefix = self.prefix_mlp(self.prefix_emb.weight)
        elif self.parameterization == "direct":
            prefix = self.prefix_direct
        else:
            raise NotImplementedError()

        batch_prefix = prefix.unsqueeze(dim=0).expand(bs, -1, -1)
        key_prefix, value_prefix = batch_prefix.chunk(dim=-1, chunks=2)
        return key_prefix, value_prefix

    def set_mode(self, mode):
        self.mode = mode
        if self.mode == "store":
            self.stored_key_value_states = None


def modify_with_prefix_tuning(transformer, config):
    transformer.config.use_cache = False
    transformer.encoder.config.use_cache = False
    transformer.decoder.config.use_cache = False

    weight_init_exist = os.path.exists(config.prefix_tuning_init_path)
    # Load saved init file
    if weight_init_exist:
        saved_model = torch.load(config.prefix_tuning_init_path)
        config.prefix_tuning_num_input_tokens = saved_model["config"]["prefix_tuning_num_input_tokens"]
        config.prefix_tuning_num_target_tokens = saved_model["config"]["prefix_tuning_num_target_tokens"]
        config.prefix_tuning_parameterization = saved_model["config"]["prefix_tuning_parameterization"]
        del saved_model["config"]
    # or Prepare init_text
    elif config.prefix_tuning_init_text != "":
        tokenizer = AutoTokenizer.from_pretrained(config.origin_model)
        input_text, target_text = config.prefix_tuning_init_text.split("\n")
        input_ids = tokenizer([input_text], add_special_tokens=False, return_tensors="pt").input_ids
        target_ids = tokenizer([target_text], add_special_tokens=False, return_tensors="pt").input_ids
        config.prefix_tuning_num_input_tokens = input_ids.size(1)
        config.prefix_tuning_num_target_tokens = target_ids.size(1)

    # Modify all attention layers
    attention_groups = [
        {
            "stack": transformer.encoder,
            "index": 0,
            "num_prefix_tokens": config.prefix_tuning_num_input_tokens,
            "module_name": "SelfAttention",
        },
        {
            "stack": transformer.decoder,
            "index": 0,
            "num_prefix_tokens": config.prefix_tuning_num_target_tokens,
            "module_name": "SelfAttention",
        },
        {
            "stack": transformer.decoder,
            "index": 1,
            "num_prefix_tokens": config.prefix_tuning_num_input_tokens,
            "module_name": "EncDecAttention",
        },
    ]

    for attention_spec in attention_groups:
        if config.prefix_tuning_parameterization.startswith("mlp"):
            hidden_size = int(config.prefix_tuning_parameterization.split("-")[1])
            shared = {
                "prefix_emb": nn.Embedding(attention_spec["num_prefix_tokens"], transformer.config.d_model),
                "prefix_linear": nn.Linear(transformer.config.d_model, hidden_size),
            }
        else:
            shared = None

        for t5_block in attention_spec["stack"].block:
            attention_layer = getattr(t5_block.layer[attention_spec["index"]], attention_spec["module_name"])
            setattr(
                t5_block.layer[attention_spec["index"]],
                attention_spec["module_name"],
                T5AttentionPrefixTuning(
                    attention_layer,
                    attention_spec["num_prefix_tokens"],
                    config.prefix_tuning_parameterization,
                    shared=shared,
                ),
            )

    # Load saved init to model
    if weight_init_exist:
        transformer.load_state_dict(saved_model, strict=False)
    # or Train prefix to approximate prefix_tuning_init_text
    elif config.prefix_tuning_init_text != "":
        # Prepare representation
        transformer.cuda()
        with torch.no_grad():
            for attention_spec in attention_groups:
                for t5_block in attention_spec["stack"].block:
                    attention_layer = getattr(t5_block.layer[attention_spec["index"]], attention_spec["module_name"])
                    attention_layer.set_mode("store")
            transformer(input_ids=input_ids.cuda(), decoder_input_ids=target_ids.cuda())
            for attention_spec in attention_groups:
                for t5_block in attention_spec["stack"].block:
                    attention_layer = getattr(t5_block.layer[attention_spec["index"]], attention_spec["module_name"])
                    attention_layer.set_mode("apply")

        # Train param
        optimizer, trainable_param_names = get_optimizer(transformer, config)
        loss_metrics = nn.MSELoss(reduction="sum")
        for step in tqdm(range(config.num_steps)):
            optimizer.zero_grad()
            list_loss = []
            for attention_spec in attention_groups:
                for t5_block in attention_spec["stack"].block:
                    attention_layer = getattr(t5_block.layer[attention_spec["index"]], attention_spec["module_name"])
                    prediction = attention_layer.get_prefix(1)
                    target = attention_layer.stored_key_value_states
                    list_loss.append(loss_metrics(prediction[0], target[0]) / (target[0] ** 2).sum())
                    list_loss.append(loss_metrics(prediction[1], target[1]) / (target[1] ** 2).sum())
            loss = sum(list_loss)
            if step % 50 == 0:
                print(step, loss.item(), [v.item() for v in list_loss])
            loss.backward()
            optimizer.step()

        trainable_states = {
            param_name: param_weight.cpu()
            for param_name, param_weight in transformer.state_dict().items()
            if param_name in trainable_param_names
        }
        if config.prefix_tuning_init_path is not None:
            trainable_states["config"] = {
                "prefix_tuning_num_input_tokens": config.prefix_tuning_num_input_tokens,
                "prefix_tuning_num_target_tokens": config.prefix_tuning_num_target_tokens,
                "prefix_tuning_parameterization": config.prefix_tuning_parameterization,
            }
            torch.save(trainable_states, config.prefix_tuning_init_path)
        transformer.cpu()

    return transformer
