CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.pl_train -c t011b.json+ia3.json+pretrain.json -k compute_strategy="ddp"  exp_name=t011b_pretrain_ia3 batch_size=4
