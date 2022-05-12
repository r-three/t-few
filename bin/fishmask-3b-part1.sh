for method in fishmask_02 fishmask_002 
do
    for dataset in copa h-swag storycloze winogrande wsc wic rte cb anli-r1 anli-r2 anli-r3
    do
        for seed in 42 1024 0 1 32
        do
            python -m src.pl_train -c t03b.json+${dataset}.json+${method}.json -k exp_name=t03b_${dataset}_seed${seed}_${method} few_shot_random_seed=${seed} seed=${seed}
        done
    done
done

