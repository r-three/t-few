for method in fishmask_02 fishmask_002 
do
    for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
    do
        for seed in 42 1024 0 1 32
        do
            python -m src.pl_train -c t03b.json+${dataset}.json+fishmask_train.json -k exp_name=t03b_${dataset}_seed${seed}_${method}_train few_shot_random_seed=${seed} seed=${seed} eval_epoch_interval=50 batch_size=4 grad_accum_factor=2 fishmask_path=exp_out/t03b_${dataset}_seed${seed}_${method}/mask.bin
        done
    done
done

