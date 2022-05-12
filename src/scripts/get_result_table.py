from glob import glob
import json
from collections import defaultdict
from scipy.stats import iqr
from numpy import median
import os
import argparse


def make_result_table(args):
    def collect_exp_scores(exp_name_template, datasets):
        print("=" * 80)
        all_files = glob(
            os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), exp_name_template, "dev_scores.json")
        )
        print(f"Find {len(all_files)} experiments fit into {exp_name_template}")

        def read_last_eval(fname):
            with open(fname) as f:
                e = json.loads(f.readlines()[-1])
            return e["accuracy"]

        acc_by_dataset = defaultdict(lambda: list())

        def parse_expname(fname):
            elements = fname.split("/")[-2].split("_")
            return tuple(elements[:3] + ["_".join(elements[3:])])

        for fname in all_files:
            result = read_last_eval(fname)
            model, dataset, seed, spec = parse_expname(fname)
            acc_by_dataset[dataset].append(result)

        def result_str(acc_list):
            if len(acc_list) > 1:
                return f"{median(acc_list) * 100:.2f} ({iqr(acc_list) * 100:.2f})"
            else:
                return f"{acc_list[0] * 100:.2f}"

        outputs = []
        for dataset in datasets:
            acc_list = acc_by_dataset[dataset]
            outputs.append(result_str(acc_list))

        print(", ".join([f"{dataset}: {value}" for dataset, value in zip(datasets, outputs)]))
        return ",".join(outputs)

    csv_lines = ["template," + (",".join(args.datasets))]
    for exp_name_template in args.exp_name_templates:
        csv_lines.append(f"{exp_name_template}," + collect_exp_scores(exp_name_template, args.datasets))

    output_fname = os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), "summary.csv")
    with open(output_fname, "w") as f:
        for line in csv_lines:
            f.write(line + "\n")
    print(f"Save result to {output_fname}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name_templates", default="t03b_*_finetune", required=True)
    parser.add_argument(
        "-d", "--datasets", default="copa,h-swag,storycloze,winogrande,wsc,wic,rte,cb,anli-r1,anli-r2,anli-r3"
    )
    args = parser.parse_args()
    args.exp_name_templates = args.exp_name_templates.split(",")
    args.datasets = args.datasets.split(",")
    make_result_table(args)
