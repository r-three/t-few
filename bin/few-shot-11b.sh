for seed in 1024 0 32 1 42
do
    for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
    do
        python -m src.pl_train -c t011b.json+ia3.json+${dataset}.json -k exp_name=t011b_${dataset}_seed${seed}_ia3 few_shot_random_seed=${seed} seed=${seed} eval_epoch_interval=50 batch_size=1 eval_batch_size=2 grad_accum_factor=8 allow_skip_exp=True
    done
done

