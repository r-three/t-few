import torch
import torch.nn as nn


def sample_embed(embed, sample_size, start_idx, end_idx):
    embed_weight = embed.weight
    rand_idx = torch.randint(start_idx, end_idx, (sample_size,))
    return embed_weight[rand_idx].detach()


class T5EncoderPromptTuningWrapper(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.num_prefix_emb = config.prompt_tuning_num_prefix_emb
        self.prompt_tuning_encoder = config.prompt_tuning_encoder
        self.encoder = encoder
        self.prompt_embedding = nn.Parameter(
            sample_embed(
                embed=encoder.get_input_embeddings(),
                sample_size=self.num_prefix_emb,
                start_idx=3,
                end_idx=5003,
            )
        )  # [num_prefix_emb, emb_dim] sampled from 5000 most common regular token embeddings

    def forward(self, input_ids, attention_mask, inputs_embeds=None, **kwargs):
        bs = input_ids.size(0)
        inputs_embeds = self.encoder.embed_tokens(input_ids)  # [bs, max_seq_len, d_emb]
        prompt_attention_mask = attention_mask.new_ones((bs, self.num_prefix_emb))  # [bs, prompt_len]
        if self.prompt_tuning_encoder:
            inputs_embeds = torch.cat(
                [self.prompt_embedding[None, :, :].repeat((bs, 1, 1)), inputs_embeds], dim=1
            )  # [bs, prompt_len+max_seq_len, d_emb]
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)  # [bs, prompt_len+max_seq_len]

        encoder_outputs = self.encoder(
            input_ids=None, inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )
        return encoder_outputs


class T5DecoderPromptTuningWrapper(nn.Module):
    def __init__(self, decoder, config):
        super().__init__()
        self.num_prefix_emb = config.prompt_tuning_num_prefix_emb
        self.prompt_tuning_encoder = config.prompt_tuning_encoder
        self.prompt_tuning_decoder = config.prompt_tuning_decoder
        self.decoder = decoder
        self.prompt_embedding = nn.Parameter(
            sample_embed(
                embed=decoder.get_input_embeddings(),
                sample_size=self.num_prefix_emb,
                start_idx=3,
                end_idx=5003,
            )
        )  # [num_prefix_emb, emb_dim] sampled from 5000 most common regular token embeddings

    def forward(self, input_ids, attention_mask, encoder_attention_mask, inputs_embeds=None, **kwargs):
        bs = input_ids.size(0)
        inputs_embeds = self.decoder.embed_tokens(input_ids)  # [bs, max_seq_len, d_emb]
        prompt_attention_mask = attention_mask.new_ones((bs, self.num_prefix_emb))  # [bs, prompt_len]
        if self.prompt_tuning_encoder:
            encoder_attention_mask = torch.cat(
                [prompt_attention_mask, encoder_attention_mask], dim=1
            )  # [bs, prompt_len+max_seq_len]
        if self.prompt_tuning_decoder:
            inputs_embeds = torch.cat(
                [self.prompt_embedding[None, :, :].repeat((bs, 1, 1)), inputs_embeds], dim=1
            )  # [bs, prompt_len+max_seq_len, d_emb]
            attention_mask = torch.cat([prompt_attention_mask, attention_mask], dim=1)  # [bs, prompt_len+max_seq_len]

        decoder_outputs = self.decoder(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        if self.prompt_tuning_decoder:
            decoder_outputs.last_hidden_state = decoder_outputs.last_hidden_state[
                :, self.num_prefix_emb :
            ]  # [bs, max_seq_len, d_emb]
        return decoder_outputs


def modify_with_prompt_tuning(transformer, config):
    transformer.encoder = T5EncoderPromptTuningWrapper(transformer.encoder, config)
    transformer.decoder = T5DecoderPromptTuningWrapper(transformer.decoder, config)
    return transformer
