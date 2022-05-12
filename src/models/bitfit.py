import torch
import torch.nn as nn
import re


def modify_with_bitfit(transformer, config):
    for m_name, module in dict(transformer.named_modules()).items():
        if re.fullmatch(config.bitfit_modules, m_name):
            for c_name, layer in dict(module.named_children()).items():
                if re.fullmatch(config.bitfit_layers, c_name):
                    layer.bias = nn.Parameter(torch.zeros(layer.out_features))
    return transformer


if __name__ == "__main__":
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    class BitFitConfig:
        def __init__(self):
            self.bitfit_modules = ".*"
            self.bitfit_layers = "q|k|v|o|w.*"
            self.trainable_param_names = ".*layer_norm.*|.*bias"
            # lora_modules and lora_layers are speicified with regular expressions
            # see https://www.w3schools.com/python/python_regex.asp for reference

    config = BitFitConfig()
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
    print(model)
    with torch.no_grad():
        old_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    model = modify_with_bitfit(model, config)

    print("New model")
    print(model)
    with torch.no_grad():
        new_outputs = model(
            input_ids=input_seq.input_ids,
            decoder_input_ids=target_seq.input_ids[:, :-1],
            labels=target_seq.input_ids[:, 1:],
        )

    print("Trainable parameters")
    print(
        [
            p_name
            for p_name in dict(model.named_parameters()).keys()
            if re.fullmatch(config.trainable_param_names, p_name)
        ]
    )

    print(f"Logits diff {torch.abs(old_outputs.logits - new_outputs.logits).mean():.3f}")
    print(f"Loss diff old={old_outputs.loss:.3f} new={new_outputs.loss:.3f}")
