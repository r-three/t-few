import os
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch.multiprocessing


from src.data import get_dataset_reader
from src.utils.util import set_seeds
from src.ticl.icl_engines import get_icl_engine
from src.ticl.model import InferenceModel

torch.multiprocessing.set_sharing_strategy("file_system")


def get_transformer(ticl_config):
    tokenizer = AutoTokenizer.from_pretrained(ticl_config.pretrained_model)
    model = AutoModelForSeq2SeqLM.from_pretrained(ticl_config.pretrained_model, low_cpu_mem_usage=True)

    tokenizer.model_max_length = ticl_config.max_seq_len
    return tokenizer, model


def main(ticl_config):
    """
    Trains the model

    :param config:
    :return:
    """

    print("Load Model")
    tokenizer, model = get_transformer(ticl_config)
    model = InferenceModel(
        model=model,
        tokenizer=tokenizer,
        length_norm=ticl_config.length_norm,
        compute_precision=ticl_config.compute_precision,
        compute_device=ticl_config.compute_device,
        compute_batch_size=ticl_config.compute_batch_size,
    )

    print("Prepare ICL Data")
    dataset_reader = get_dataset_reader(ticl_config)
    set_seeds(ticl_config.a_number_between_1_and_100)
    icl_engine = get_icl_engine(ticl_config, dataset_reader, tokenizer)

    print("Start Eval")
    icl_engine.run(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="debug")

    parser.add_argument("--pretrained_model", type=str, default="google/t5-xxl-lm-adapt")
    parser.add_argument("--a_number_between_1_and_100", type=int, default=62)
    parser.add_argument("--length_norm", type=str, choices=["True", "False"], default="True")
    parser.add_argument("--icl_method", type=str, choices=["concat", "ensemble"], default="ensemble")
    parser.add_argument("--icl_modeling", type=str, choices=["direct", "calibrated", "channel"], default="direct")

    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--num_shot", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--train_template_idx", type=int, default=-1)
    parser.add_argument("--eval_template_idx", type=int, default=-1)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--max_context_len", type=int, default=8192)
    parser.add_argument("--context_start", type=str, default="")
    parser.add_argument("--context_input_target_separator", type=str, default="")
    parser.add_argument("--context_example_separator", type=str, default="")
    parser.add_argument("--context_end", type=str, default="")

    parser.add_argument("--compute_precision", type=str, choices=["16", "32", "bf16"], default="bf16")
    parser.add_argument("--compute_device", type=str, choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--compute_batch_size", type=int, default=128)  # Not used at present

    ticl_config = parser.parse_args()

    ticl_config.length_norm = ticl_config.length_norm == "True"
    ticl_config.exp_dir = os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), ticl_config.exp_name)
    ticl_config.dev_pred_file = os.path.join(ticl_config.exp_dir, "dev_pred.txt")
    ticl_config.dev_score_file = os.path.join(ticl_config.exp_dir, "dev_scores.json")
    ticl_config.finish_flag_file = os.path.join(ticl_config.exp_dir, "exp_completed.txt")
    ticl_config.change_hswag_templates = False
    if not os.path.exists(ticl_config.exp_dir):
        os.makedirs(ticl_config.exp_dir)

    print(ticl_config)

    if os.path.exists(ticl_config.finish_flag_file):
        print(f"Skip finished experiment {ticl_config.exp_name}")
    else:
        with open(os.path.join(ticl_config.exp_dir, os.path.join("config.txt")), "w") as f:
            f.write(ticl_config.__str__())

        print(f"Mark experiment {ticl_config.exp_name} as claimed")
        with open(ticl_config.finish_flag_file, "w") as f:
            f.write("0")
        main(ticl_config)
