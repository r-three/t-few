#!/usr/bin/env bash

set -exu

seed=$1
model=$2
method=$3
dataset=$4
template_idx=$5
descriptor=$6

export orig_exp_name=${model}_${dataset}_seed${seed}_${method}_${descriptor}
python -m src.pl_train -c ${model}.json+${method}.json+${dataset}.json -k load_weight=exp_out/${orig_exp_name}/finish.pt save_model=False exp_name=${orig_exp_name}_template_${template_idx} few_shot_random_seed=${seed} seed=${seed} compute_strategy="none" save_model=False allow_skip_exp=True num_steps=0 eval_template_idx=${template_idx}
