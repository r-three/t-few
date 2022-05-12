import torch.nn.functional as F
import torch


class InferenceModel:
    def __init__(self, model, tokenizer, length_norm, compute_precision, compute_device, compute_batch_size):
        self.model = model
        self.tokenizer = tokenizer
        self.length_norm = length_norm
        self.compute_precision = compute_precision
        self.compute_device = compute_device
        self.compute_batch_size = compute_batch_size
        self.model.eval()
        self._model_to_device()

    def __call__(self, batch):
        with torch.no_grad():
            input_ids, choices_ids = batch["input_ids"], batch["answer_choices_ids"]
            input_ids = input_ids.to(self.model.encoder.device)
            choices_ids = choices_ids.to(self.model.decoder.device)

            # TODO: use compute_batch_size if CUDA OOM
            bs, num_choices = choices_ids.size()[:2]
            flat_input_ids = input_ids.flatten(0, 1)
            choices_sharing_inputs = input_ids.size(1) != num_choices

            attention_mask = (flat_input_ids != self.tokenizer.pad_token_id).float()
            # (bs * (1 or num_choices), max_seq_len)
            encoder_hidden_states = self.model.encoder(input_ids=flat_input_ids, attention_mask=attention_mask)[0]
            # (bs * (1 or num_choices), max_seq_len, d_model)
            if choices_sharing_inputs:
                attention_mask = attention_mask.unsqueeze(dim=1).expand(-1, num_choices, -1).flatten(0, 1)
                # (bs * num_choices, max_seq_len)
                encoder_hidden_states = (
                    encoder_hidden_states.unsqueeze(dim=1).expand(-1, num_choices, -1, -1).flatten(0, 1)
                )
                # (bs * num_choices, max_seq_len, d_model)

            flat_choices_ids = choices_ids.flatten(0, 1)
            decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
            lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()
            # (bs * num_choices, max_seq_len)

            model_output = self.model(
                attention_mask=attention_mask,
                encoder_outputs=[encoder_hidden_states],
                decoder_input_ids=decoder_input_ids,
            )
            choices_scores = (
                F.cross_entropy(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                .view(bs, num_choices, -1)
                .sum(dim=-1)
            )
            if self.length_norm:
                choices_scores = choices_scores / (choices_ids != self.tokenizer.pad_token_id).sum(dim=-1)
            # (bs, num_choices)

        return choices_scores.cpu()

    def _model_to_device(self):
        if self.compute_precision == "32":
            pass
        elif self.compute_precision == "16":
            self.model.to(torch.float16)
        elif self.compute_precision == "bf16":
            self.model.to(torch.bfloat16)

        self.model.to(self.compute_device)
        if self.compute_device == "cuda" and torch.cuda.device_count() > 1:
            self.model.parallelize()
