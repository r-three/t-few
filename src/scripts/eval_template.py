import subprocess
import argparse

dict_dataset_2_template_idx = {
    "copa": list(range(12)),
    "h-swag": [3, 4, 8, 10],
    "storycloze": [0, 1, 3, 4, 5],
    "winogrande": [0, 2, 3, 4, 5],
    "wsc": list(range(10)),
    "wic": list(range(10)),
    "rte": list(range(10)),
    "cb": list(range(15)),
    "anli-r1": list(range(15)),
    "anli-r2": list(range(15)),
    "anli-r3": list(range(15)),
}


def eval_random_template(model, method, descriptor):

    for seed in [0, 1, 32, 42, 1024]:
        for dataset in ["copa", "h-swag", "storycloze", "winogrande", "wic", "wsc", "rte", "cb", "anli-r1", "anli-r2", "anli-r3"]:
            if descriptor is None:
                command = f"bash bin/eval-template.sh {seed} {model} {method} {dataset} -1"
            else:
                command = f"bash bin/eval-template-with-descriptor.sh {seed} {model} {method} {dataset} -1 {descriptor}"
            subprocess.run([command], stdout=subprocess.PIPE, shell=True)


def eval_all_templates(model, method, descriptor):

    for seed in [0, 1, 32, 42, 1024]:
        for dataset in ["copa", "h-swag", "storycloze", "winogrande", "wic", "wsc", "rte", "cb", "anli-r1", "anli-r2", "anli-r3"]:
            for template_idx in dict_dataset_2_template_idx[dataset]:
                if descriptor is None:
                    command = f"bash bin/eval-template.sh {seed} {model} {method} {dataset} {template_idx}"
                else:
                    command = f"bash bin/eval-template-with-descriptor.sh {seed} {model} {method} {dataset} {template_idx} {descriptor}"
                subprocess.run([command], stdout=subprocess.PIPE, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all_template_or_random_template", required=True, choices=["all", "random"])
    parser.add_argument("-model", "--model", required=True)
    parser.add_argument("-method", "--method", required=True)
    parser.add_argument("-d", "--descriptor", default=None)
    args = parser.parse_args()


    if args.all_template_or_random_template == "all":
        eval_all_templates(args.model, args.method, args.descriptor)
    else:
        eval_random_template(args.model, args.method, args.descriptor)

