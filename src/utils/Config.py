import json
import os
import ast


class Config(object):
    def __init__(self, filenames=None, kwargs=None):
        # Experiment configs
        self.exp_dir = None
        self.exp_name = None
        self.allow_skip_exp = True
        self.seed = 42

        # Model Configs
        self.model = "EncDec"
        self.max_seq_len = 256
        self.origin_model = "bigscience/T0_3B"
        self.load_weight = ""

        # Dataset Configs
        self.dataset = "rte"
        self.few_shot = True
        self.num_shot = 16
        self.few_shot_random_seed = 100
        self.train_template_idx = -1
        self.eval_template_idx = -1
        self.batch_size = 8
        self.eval_batch_size = 16
        self.num_workers = 8
        self.change_hswag_templates = False
        self.raft_cross_validation = True
        self.raft_validation_start = 0
        self.raft_labels_in_input_string = "comma"
        self.cleaned_answer_choices_b77 = False

        # Compute backend configs
        self.compute_precision = "bf16"
        self.compute_strategy = "none"

        # Trainer configs
        self.num_steps = 300
        self.eval_epoch_interval = 10_000
        self.eval_before_training = True
        self.save_model = True
        self.save_step_interval = 20_000
        self.mc_loss = 0
        self.unlikely_loss = 0
        self.length_norm = 0
        self.grad_accum_factor = 1
        self.split_option_at_inference = False  # Whether to split the answer choices during eval to lower memory usage for datasets with lots of answer choices

        # Optimization configs
        self.optimizer = "adafactor"
        self.lr = 3e-4
        self.trainable_param_names = ".*"
        self.scheduler = "linear_decay_with_warmup"
        self.warmup_ratio = 0.06
        self.weight_decay = 0
        self.scale_parameter = True
        self.grad_clip_norm = 1

        # PEFT method configs
        self.model_modifier = ""
        # Prompt Tuning configs
        self.prompt_tuning_num_prefix_emb = 100
        self.prompt_tuning_encoder = True
        self.prompt_tuning_decoder = True
        # LoRA configs
        self.lora_rank = 4
        self.lora_scaling_rank = 0
        self.lora_init_scale = 0.01
        self.lora_modules = "none"
        self.lora_layers = "none"
        # BitFit configs
        self.bitfit_modules = ".*"
        self.bitfit_layers = "q|k|v|o|wi_[01]|w_o"
        # Adapter configs
        self.adapter_type = "normal"
        self.adapter_non_linearity = "relu"
        self.adapter_reduction_factor = 4
        self.normal_adapter_residual = True
        self.lowrank_adapter_w_init = "glorot-uniform"
        self.lowrank_adapter_rank = 1
        self.compacter_hypercomplex_division = 8
        self.compacter_learn_phm = True
        self.compacter_hypercomplex_nonlinearity = "glorot-uniform"  # wait, is this really the right name?
        self.compacter_shared_phm_rule = False
        self.compacter_factorized_phm = False
        self.compacter_shared_W_phm = False
        self.compacter_factorized_phm_rule = False
        self.compacter_phm_c_init = "normal"
        self.compacter_phm_rank = 1
        self.compacter_phm_init_range = 0.01
        self.compacter_kronecker_prod = False
        self.compacter_add_compacter_in_self_attention = False
        self.compacter_add_compacter_in_cross_attention = False
        # Intrinsic SAID configs
        self.intrinsic_projection = "fastfood"
        self.intrinsic_said = True
        self.intrinsic_dim = 2000
        self.intrinsic_device = "cpu"
        # FISH mask configs
        self.fishmask_mode = None
        self.fishmask_path = None
        self.fishmask_keep_ratio = 0.05
        # Prefix Tuning configs
        self.prefix_tuning_num_input_tokens = 10
        self.prefix_tuning_num_target_tokens = 10
        self.prefix_tuning_init_path = None
        self.prefix_tuning_init_text = None
        self.prefix_tuning_parameterization = "mlp-512"

        if filenames:
            for filename in filenames.split("+"):
                if not os.path.exists(filename):
                    filename = os.path.join(os.getenv("CONFIG_PATH", default="configs"), filename)

                self.update_kwargs(json.load(open(filename)), eval=False)
        if kwargs:
            self.update_kwargs(kwargs)

        self.set_exp_dir()

    def update_kwargs(self, kwargs, eval=True):
        for (k, v) in kwargs.items():
            if eval:
                try:
                    v = ast.literal_eval(v)
                except ValueError:
                    v = v
            else:
                v = v
            if not hasattr(self, k):
                raise ValueError(f"{k} is not in the config")
            setattr(self, k, v)

    def set_exp_dir(self):
        """
        Updates the config default values based on parameters passed in from config file
        """

        if self.exp_name is not None:
            self.exp_dir = os.path.join(os.getenv("OUTPUT_PATH", default="exp_out"), self.exp_name)
        else:
            self.exp_dir = os.getenv("OUTPUT_PATH", default="exp_out")
        if not os.path.exists(self.exp_dir):
            os.makedirs(self.exp_dir)

        if self.exp_dir is not None:
            self.train_pred_file = os.path.join(self.exp_dir, "train_pred.txt")
            self.dev_pred_file = os.path.join(self.exp_dir, "dev_pred.txt")
            self.dev_score_file = os.path.join(self.exp_dir, "dev_scores.json")
            self.test_pred_file = os.path.join(self.exp_dir, "test_pred.txt")
            self.test_score_file = os.path.join(self.exp_dir, "test_scores.json")
            self.save_config(os.path.join(self.exp_dir, os.path.join("config.json")))
            self.finish_flag_file = os.path.join(self.exp_dir, "exp_completed.txt")

    def to_json(self):
        """
        Converts parameter values in config to json
        :return: json
        """
        return json.dumps(self.__dict__, indent=4, sort_keys=False)

    def save_config(self, filename):
        """
        Saves the config
        """
        with open(filename, "w+") as fout:
            fout.write(self.to_json())
            fout.write("\n")
