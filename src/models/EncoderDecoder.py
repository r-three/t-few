import os
import json
import torch
import torch.nn.functional as F
import torch.distributed as dist
from pytorch_lightning import LightningModule
from src.utils.get_optimizer import get_optimizer
from src.utils.get_scheduler import get_scheduler
from statistics import mean
from deepspeed.utils import zero_to_fp32
from .fishmask import fishmask_plugin_on_init, fishmask_plugin_on_optimizer_step, fishmask_plugin_on_end


class EncoderDecoder(LightningModule):
    """
    Encoder Decoder
    """

    def __init__(self, config, tokenizer, transformer, dataset_reader):
        """
        :param config
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.model = transformer
        self.dataset_reader = dataset_reader

        self.use_deepspeed = self.config.compute_strategy.startswith("deepspeed")
        self.use_ddp = self.config.compute_strategy.startswith("ddp")
        self.load_model()

        self._last_global_step_saved = -1

        if self.config.fishmask_mode is not None:
            fishmask_plugin_on_init(self)

    def training_step(self, batch, batch_idx):
        if self.config.model_modifier == "intrinsic":
            from .intrinsic import intrinsic_plugin_on_step
            intrinsic_plugin_on_step(self)

        if self.config.mc_loss > 0 or self.config.unlikely_loss > 0:
            input_ids, choices_ids, labels = batch["input_ids"], batch["answer_choices_ids"], batch["labels"]
            bs, num_choices = choices_ids.size()[:2]

            flat_choices_ids = choices_ids.flatten(0, 1)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
            encoder_hidden_states = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)
            attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)
            decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
            decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
            lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()

            model_output = self.model(
                attention_mask=attention_mask,
                encoder_outputs=[encoder_hidden_states],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            choices_scores = (
                F.cross_entropy(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                .view(bs, num_choices, -1)
                .sum(dim=-1)
            )
            if self.config.length_norm > 0:
                choices_scores = choices_scores / torch.pow(
                    (choices_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
                )
            lm_loss = F.cross_entropy(
                model_output.logits.view(bs, num_choices, *model_output.logits.size()[1:])[range(bs), labels].flatten(
                    0, 1
                ),
                lm_target.view(bs, num_choices, -1)[range(bs), labels].flatten(0, 1),
            )

            tensorboard_logs = {"lm_loss": lm_loss.item()}
            if self.config.mc_loss > 0:
                mc_loss = F.cross_entropy(-choices_scores, labels)
                tensorboard_logs["mc_loss"] = mc_loss.item()
            else:
                mc_loss = 0.0

            if self.config.unlikely_loss > 0:
                cand_loglikely = -F.cross_entropy(
                    model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none"
                ).view(bs, num_choices, -1)
                cand_loglikely += (lm_target < 0).view(bs, num_choices, -1) * -100
                cand_loglikely[range(bs), labels] = -100
                unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != -100).sum()
                tensorboard_logs["unlikely_loss"] = unlikely_loss.item()
            else:
                unlikely_loss = 0.0

            loss = lm_loss + mc_loss * self.config.mc_loss + unlikely_loss * self.config.unlikely_loss
            tensorboard_logs["loss"] = loss.item()
        else:
            input_ids, target_ids = batch["input_ids"], batch["target_ids"]
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
            lm_labels = target_ids + -100 * (target_ids == self.tokenizer.pad_token_id).long()  # [bs, max_seq_len]
            decoder_input_ids = torch.cat(
                [torch.zeros_like(lm_labels[:, :1]), target_ids[:, :-1]], dim=1
            )  # [bs, max_seq_len]
            decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()

            model_output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=lm_labels,
            )
            loss = model_output.loss
            tensorboard_logs = {"loss": loss.item()}

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            self.log_dict(tensorboard_logs)

        if self.global_step % self.config.save_step_interval == 0:
            self.save_model()

        return loss

    def predict(self, batch):
        """
        Predict the lbl for particular pet
        :param batch:
        :param pet:
        :return:
        """
        if self.config.model_modifier == "intrinsic":
            from .intrinsic import intrinsic_plugin_on_step
            intrinsic_plugin_on_step(self)

        input_ids, choices_ids, labels = batch["input_ids"], batch["answer_choices_ids"], batch["labels"]

        if not self.config.split_option_at_inference:
            bs, num_choices = choices_ids.size()[:2]
            flat_choices_ids = choices_ids.flatten(0, 1)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
            encoder_hidden_states = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)
            attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)
            decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
            decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
            lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()

            model_output = self.model(
                attention_mask=attention_mask,
                encoder_outputs=[encoder_hidden_states],
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
            )
            choices_scores = (
                F.cross_entropy(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                .view(bs, num_choices, -1)
                .sum(dim=-1)
            )
            if self.config.length_norm > 0:
                choices_scores = choices_scores / torch.pow(
                    (choices_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
                )
            pred_score, prediction = choices_scores.min(dim=1)

        else:
            bs, num_choices = choices_ids.size()[:2]
            midpoint = num_choices // 2
            #
            first_half_choice_ids = choices_ids[:, :midpoint, :]
            second_half_choice_ids = choices_ids[:, midpoint:, :]
            #
            all_choice_scores = []

            for half_choice_ids in [first_half_choice_ids, second_half_choice_ids]:
                half_num_choices = half_choice_ids.shape[1]

                flat_choices_ids = half_choice_ids.flatten(0, 1)  # [bs*num_choices, choice_len]

                attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
                encoder_hidden_states = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
                encoder_hidden_states = (
                    encoder_hidden_states.unsqueeze(dim=1).repeat(1, half_num_choices, 1, 1).flatten(0, 1)
                )
                attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, half_num_choices, 1).flatten(0, 1)

                decoder_input_ids = torch.cat(
                    [torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1
                )
                decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
                lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()

                model_output = self.model(
                    attention_mask=attention_mask,
                    encoder_outputs=[encoder_hidden_states],
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )
                choices_scores = (
                    F.cross_entropy(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
                    .view(bs, half_num_choices, -1)
                    .sum(dim=-1)
                )
                if self.config.length_norm > 0:
                    choices_scores = choices_scores / torch.pow(
                        (half_choice_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.config.length_norm
                    )

                all_choice_scores.append(choices_scores)

            choices_scores = torch.cat(all_choice_scores, dim=-1)
            pred_score, prediction = choices_scores.min(dim=1)

        score_gt = choices_scores[range(bs), labels]
        choices_scores[range(bs), labels] = choices_scores.max(dim=-1)[0]
        score_cand = choices_scores.min(dim=-1)[0]

        batch_output = {
            "prediction": prediction.tolist(),
            "label": labels.tolist(),
            "idx": batch["idx"].tolist(),
            "log.score_gt": score_gt.tolist(),
            "log.score_cand": score_cand.tolist(),
        }
        return batch_output

    def validation_step(self, batch, batch_idx):
        batch_output = self.predict(batch)
        return batch_output

    def validation_epoch_end(self, outputs):
        # exchange outputs between processes
        if self.use_deepspeed or self.use_ddp:
            gathered_outputs = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_outputs, outputs)
            if dist.get_rank() == 0:
                outputs = [batch_output for outputs in gathered_outputs for batch_output in outputs]

        if not (self.use_deepspeed or self.use_ddp) or dist.get_rank() == 0:
            # let rank 0 collect all outputs
            accumulated = {key: [] for key in outputs[0].keys()}
            for batch_output in outputs:
                for key, value in batch_output.items():
                    accumulated[key].extend(value)

            # multi-process may yield dupliated examples in the last batch
            valid_mask = []
            idx_set = set()
            for idx in accumulated["idx"]:
                valid_mask.append(idx not in idx_set)
                idx_set.add(idx)
            for key, values in accumulated.items():
                accumulated[key] = [v for v, m in zip(values, valid_mask) if m]

            # compute and log results
            metrics = self.dataset_reader.compute_metric(accumulated)
            for key, value in accumulated.items():
                if key.startswith("log."):
                    metrics[key.replace("log.", "")] = mean(value)

            result_str = json.dumps(metrics) + "\n"
            with open(self.config.dev_score_file, "a+") as f:
                f.write(result_str)
            print("\n" + result_str)
        else:
            metrics = {}

        self.save_model()
        return metrics

    def configure_optimizers(self):
        optimizer, self.trainable_param_names = get_optimizer(self.model, self.config)
        scheduler = get_scheduler(optimizer, self.config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def on_train_end(self):
        self.save_model(finish=True)

        if self.config.fishmask_mode is not None:
            fishmask_plugin_on_end(self)

    def load_model(self):
        if self.config.load_weight != "":
            trainable_states = torch.load(self.config.load_weight, map_location=torch.device("cpu"))
            load_result = self.model.load_state_dict(trainable_states, strict=False)
            assert (
                len(load_result.unexpected_keys) == 0
            ), f"Load model failed, unexpected keys {load_result.unexpected_keys.__str__()}"

    def save_model(self, finish=False):
        if self.config.save_model and (finish or self._last_global_step_saved != self.global_step):
            if finish:
                model_fname = os.path.join(self.config.exp_dir, "finish.pt")
            else:
                model_fname = os.path.join(self.config.exp_dir, f"global_step{self.global_step}.pt")

            if self.use_deepspeed or self.use_ddp:
                distributed_save_path = os.path.join(self.config.exp_dir, "saved_model")
                self.trainer.model.save_checkpoint(distributed_save_path)
                torch.distributed.barrier()
                if dist.get_rank() == 0:
                    trainable_states = zero_to_fp32.get_fp32_state_dict_from_zero_checkpoint(distributed_save_path)
                    prefix_length = len("module.model.")
                    trainable_states = {k[prefix_length:]: v for k, v in trainable_states.items()}
                    torch.save(trainable_states, model_fname)
            else:
                trainable_states = {
                    param_name: param_weight.cpu()
                    for param_name, param_weight in self.model.state_dict().items()
                    if param_name in self.trainable_param_names
                }
                torch.save(trainable_states, model_fname)

            self._last_global_step_saved = self.global_step

    def on_before_optimizer_step(self, optimizer, optimizer_idx):
        if self.config.fishmask_mode is not None:
            fishmask_plugin_on_optimizer_step(self)
