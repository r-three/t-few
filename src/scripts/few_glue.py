
import seqio
import os
import json
import functools

import tensorflow as tf

from prompt_tuning.data import features
from prompt_tuning.data import metrics as pt_metrics
from prompt_tuning.data import postprocessors as pt_postprocessors
from prompt_tuning.data import preprocessors as pt_preprocessors
from prompt_tuning.data import utils
import seqio
from t5.data.glue_utils import get_glue_metric
from t5.data.glue_utils import get_glue_postprocess_fn
from t5.data.glue_utils import get_glue_text_preprocessor
from t5.data.glue_utils import get_glue_weight_mapping
import tensorflow_datasets as tfds

import t5.data.tasks as t5_tasks

# Aliases for ease of use when you want features for a specific model.
T5_FEATURES = t5_tasks.DEFAULT_OUTPUT_FEATURES

glue_indexer = utils.task_mapping(
    tuple(b.name for b in tfds.text.glue.Glue.builder_configs.values()),
    {
        "ax": "mnli",
        "mnli_mismatched": "mnli",
        "mnli_matched": "mnli"
    },
)


DATASET_KEY_MAPPINGS = {
    "rte": {"premise": "sentence1",
            "hypothesis": "sentence2",
            "idx": "idx",
            "label": "label"}
}


DATASET_OUTPUT_SIGNATURES = {
    "rte": {"sentence1": tf.TensorSpec(shape=(), dtype=tf.string),
            "sentence2": tf.TensorSpec(shape=(), dtype=tf.string),
            "idx": tf.TensorSpec(shape=(), dtype=tf.int32),
            "label": tf.TensorSpec(shape=(), dtype=tf.int64)}
}


def read_dataset(dataset_name, num_shot, few_shot_seed):
    '''
    Return dataset as list of json dictionaries

    :param dataset_name: name of dataset 
    :param num_shot: number of examples per class 
    :param few_shot_seed: seed used to sample few-shot examples 
    '''
    fp = os.path.join("data", "few_shot", dataset_name, f"{num_shot}_shot", f"{few_shot_seed}_seed.jsonl")
    print("fp: ", fp)
    key_mappings = DATASET_KEY_MAPPINGS[dataset_name]

    with open(fp, "r") as fin:
        data = []
        for idx, line in enumerate(fin.readlines()):
            original_json = json.loads(line.strip("\n"))
            print(original_json)
            updated_json = {}
            for (k, v) in original_json.items():
                updated_json[key_mappings[k]] = v   
            yield updated_json
    


list_datasets = ["rte"]
list_num_shot = [32]
list_few_shot_seed = [32]





for dataset_name in list_datasets:

    builder_config = tfds.text.glue.Glue.builder_configs[dataset_name]
    postprocess_fn = get_glue_postprocess_fn(builder_config)
    metric_fns = get_glue_metric(builder_config.name)

    validation_ds = seqio.TfdsDataSource(tfds_name=f"glue/{builder_config.name}:2.0.0").get_dataset("validation")

    for num_shot in list_num_shot:
        for few_shot_seed in list_few_shot_seed:

            train_ds = tf.data.Dataset.from_generator(
                functools.partial(read_dataset, dataset_name=dataset_name, num_shot=num_shot, few_shot_seed=few_shot_seed), output_signature=DATASET_OUTPUT_SIGNATURES[dataset_name]
            )

            def dataset_fn(split, shuffle_files, seed):
                if split == "train":
                    return train_ds
                elif split == "validation":
                    return validation_ds
                else:
                    raise ValueError(f"Invalid split {split}")
            
            data_source = seqio.FunctionDataSource(
                dataset_fn,
                splits=["train", "validation"],
                num_input_examples={"train": num_shot, "validation": len(validation_ds)},
            )
            # Has task index as first token 
            seqio.TaskRegistry.add(
                f"glue_{dataset_name}_{num_shot}_shot_{few_shot_seed}_seed",
                source=data_source,
                preprocessors=[
                    get_glue_text_preprocessor(builder_config),
                    pt_preprocessors.remove_first_text_token,
                    seqio.preprocessors.tokenize,
                    functools.partial(
                        pt_preprocessors.add_sentinel_to_beginning,
                        field="inputs",
                        offset=glue_indexer[builder_config.name]),
                    seqio.CacheDatasetPlaceholder(),
                    seqio.preprocessors.append_eos_after_trim,
                ],
                postprocess_fn=postprocess_fn,
                metric_fns=metric_fns,
                output_features=T5_FEATURES)

