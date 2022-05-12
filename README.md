# T-Few

This repository contains the official code for the paper: "[Few-Shot Parameter-Efficient Fine-Tuning Outperforms In-Context Learning]()".

This method outperforms in-context learning with GPT-3 and achieves state-of-the-art on "[RAFT](https://huggingface.co/spaces/ought/raft-leaderboard)".

## Setup

First, create a virtual environment for the project and install all the requirments.
(We use conda to manage environments. Be sure to install and initialize conda first.)

1. Create a virtual environment with python 3.7 `conda create -n tfew python==3.7`, then activate the environment `conda activate tfew`.
2. Download NICL repo, switch to `t0` branch, and install other dependencies. `pip install -r requirements.txt -f https://download.pytorch.org/whl/cu113/torch_stable.html`
3. Run `python src/intrinsic_said_setup.py develop`

The steps above only needs to be done once. In addition, every time you start a new session, you will need to run `. bin/start.sh`

## Run your first experiment

Once you finished setting up the environment, you can try running
`CUDA_VISIBLE_DEVICES=3 python -m src.pl_train -c t0.json+rte.json -k save_model=False exp_name=first_exp`
The outputs of this run will be saved to `${OUTPUT_PATH}/first_exp/`, which is usually `/nicl/exp_out/first_exp/`. Here, `first_exp` is the experiment name, you can run more experiments with different expeirment names. The code will automatically skip finished experiments. (However, if you wish to rerun a finished experiment under the same experiment name, you will need to manually remove the corresponding files in the output directory.)

There are two ways to control an experiment.

1. You can specify config files with `-c`. Multiple config files can be combined with `+`. (When there are conflits, config terms from the config file on the right will have greater power.) This will be convinient when you have multiple terms that forms a fixed group.
2. You can override values with `-k`. This will be convinient when you need to change a small number of terms.

It is recommended to use GPUs with 40GB to train T0(3B) and 80GB to train T0(11B)

## Run an array of experiments

In this project, we often need to run a large number of experiments.
Here is an example bash script `bin/few-shot-pretrained-3b-100k.sh` to fine-tune 3B pre-trained (IA)3 on all datasets.

This should take a few hours. After that, you can use `scripts/get_results_table.py` to generate a csv summary.

## Citation

We use the following code in our works:

```
@article{mahabadi2021compacter,
  title={Compacter: Efficient low-rank hypercomplex adapter layers},
  author={Mahabadi, Rabeeh Karimi and Henderson, James and Ruder, Sebastian},
  journal={arXiv preprint arXiv:2106.04647},
  year={2021}
}

@article{sung2021training,
  title={Training Neural Networks with Fixed Sparse Masks},
  author={Sung, Yi-Lin and Nair, Varun and Raffel, Colin},
  journal={arXiv preprint arXiv:2111.09839},
  year={2021}
}
```
