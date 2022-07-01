import os
from typing import Union
import torch
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5TokenizerFast
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import FinetuneDataModule, get_dataset_reader, PretrainDataModule
from src.data.data_module import FinetuneDatasetWithTemplate, create_collate_fn
from src.models.EncoderDecoder import EncoderDecoder
from src.models.modify_model import modify_transformer
from src.utils.Config import Config
from src.utils.util import ParseKwargs, set_seeds
from pytorch_lightning.plugins import DeepSpeedPlugin
import numpy as np


def deep_speed_strategy(config: Config) -> DeepSpeedPlugin:
    return DeepSpeedPlugin(
        stage=config.ds_stage,
        offload_optimizer=config.ds_offload_optimizer,
        cpu_checkpointing=config.ds_cpu_checkpointing,
        offload_parameters=config.ds_offload_params,
        remote_device="nvme" if config.ds_nvme else "cpu",
        offload_params_device="nvme" if config.ds_nvme else "cpu",
        offload_optimizer_device="nvme" if config.ds_nvme else "cpu",
        logging_batch_size_per_gpu=config.batch_size,
    )


def get_transformer(config: Config):
    tokenizer: T5TokenizerFast = AutoTokenizer.from_pretrained(config.origin_model)
    model: T5ForConditionalGeneration = AutoModelForSeq2SeqLM.from_pretrained(config.origin_model, low_cpu_mem_usage=True)

    tokenizer.model_max_length = config.max_seq_len
    model = modify_transformer(model, config)
    return tokenizer, model


def infer_batch(model, dataset, collate_fn, start, stop):
    batch = collate_fn([dataset[i] for i in range(start, stop)])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        return model.predict(batch)

def infer_all_batches(model, dataset, collate_fn, batch_size: int = 4):
    ranges = [range(i, i + batch_size) for i in range(0, len(dataset), batch_size)]
    for val in ranges:
        batch = collate_fn([dataset[i] for i in val if i < len(dataset)])
        with torch.no_grad():
            yield model.predict(batch)


def score_all_batches(model, dataset: FinetuneDatasetWithTemplate, collate_fn, batch_size: int = 4):
    batches = infer_all_batches(model, dataset, collate_fn, batch_size)
    predictions: list[int] = []
    labels: list[int] = []
    for batch in batches:
        predictions.extend(batch["prediction"])
        labels.extend(batch["label"])
    predictions = np.array(predictions)
    labels = np.array(labels)
    TP = np.sum(predictions * labels)
    FP = np.sum(predictions * (1 - labels))
    TN = np.sum((1 - predictions) * (1 - labels))
    FN = np.sum((1 - predictions) * labels)
    TP_idxs = np.where(predictions * labels)[0]
    FP_idxs = np.where(predictions * (1 - labels))[0]
    FN_idxs = np.where((1 - predictions) * labels)[0]
    tp_items = [dataset.dataset[i] for i in TP_idxs]
    fp_items = [dataset.dataset[i] for i in FP_idxs]
    fn_items = [dataset.dataset[i] for i in FN_idxs]
    print(f"TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}")
    return tp_items, fp_items, fn_items, predictions, labels
    

def load_for_inference(config: Union[Config, str]):

    if isinstance(config, str):
        config = Config(config)

    tokenizer, model = get_transformer(config)
    dataset_reader = get_dataset_reader(config)
    model = EncoderDecoder(config, tokenizer, model, dataset_reader)
    model.config.load_weight = os.path.join(config.exp_dir, "finish.pt")
    model.load_model()
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    collate_fn = create_collate_fn(tokenizer.pad_token_id, pretrain=False)

    val_orig_dataset = dataset_reader.read_orig_dataset("validation")
    val_dataset = FinetuneDatasetWithTemplate(val_orig_dataset, dataset_reader.get_eval_template(), tokenizer)

    return model, val_dataset, collate_fn

def predict_range(model: EncoderDecoder, val_dataset, collate_fn, start, stop):
    batch = collate_fn([val_dataset[i] for i in range(start, stop)])
    batch = {k: v.to(model.device) for k, v in batch.items()}
    with torch.no_grad():
        return model.predict(batch)


def main(config: Config):
    """
    Trains the model

    :param config:
    :return:
    """

    tokenizer, model = get_transformer(config)
    dataset_reader = get_dataset_reader(config)
    if config.dataset == "T0Mixture":
        datamodule = PretrainDataModule(config, tokenizer, dataset_reader)
    else:
        datamodule = FinetuneDataModule(config, tokenizer, dataset_reader)
    model = EncoderDecoder(config, tokenizer, model, dataset_reader)
    breakpoint()
    logger = TensorBoardLogger(config.exp_dir, name="log")

    if "deepspeed" in config.compute_strategy or config.use_deepspeed:
        strategy = deep_speed_strategy(config)
        print("Using DeepSpeed lightning plugin")
    else:
        strategy = config.compute_strategy if config.compute_strategy != "none" else None

    torch.cuda.empty_cache()

    trainer = Trainer(
        enable_checkpointing=False,
        gpus=torch.cuda.device_count(),
        precision=config.compute_precision,
        amp_backend="native",
        strategy=strategy,
        logger=logger,
        log_every_n_steps=4,
        max_steps=config.num_steps,
        min_steps=config.num_steps,
        num_sanity_val_steps=-1 if config.eval_before_training else 0,
        val_check_interval=config.eval_step_interval,
        accumulate_grad_batches=config.grad_accum_factor,
        gradient_clip_val=config.grad_clip_norm,
    )
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_files", required=True)
    parser.add_argument("-k", "--kwargs", nargs="*", action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_files, args.kwargs)
    print(f"Start experiment {config.exp_name}")
    # Setup config
    assert config.compute_strategy in ["none", "ddp", "deepspeed_stage_3_offload", "deepspeed_stage_3"]
    if config.fishmask_mode == "create":
        print("Detecting fishmask_mode=create, override batch_size, num_step, fishmask_path")
        config.batch_size = 1
        config.num_steps = config.num_shot
        config.eval_before_training = False
        config.fishmask_path = None

    print(config.to_json())

    if config.allow_skip_exp and os.path.exists(config.finish_flag_file):
        print(f"Skip finished experiment {config.exp_name}")
    else:
        print(f"Mark experiment {config.exp_name} as claimed")
        with open(config.finish_flag_file, "a+") as f:
            f.write(datetime.now().strftime("%m/%d/%Y %H:%M:%S") + "\n")
        set_seeds(config.seed)
        main(config)
