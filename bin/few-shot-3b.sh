for method in ia3 lora lora_scaling bitfit layernorm compacter compacter_pp prompt_tuning-unlikely_100_prompts prompt_tuning-unlikely_10_prompts adapter intrinsic_said_20k intrinsic_said_500k prefix_tuning
do
    for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
    do
        for seed in 42 1024 0 1 32
        do
            python -m src.pl_train -c t03b.json+${dataset}.json+${method}.json -k exp_name=t03b_${dataset}_seed${seed}_${method} few_shot_random_seed=${seed} seed=${seed}
        done
    done
done


