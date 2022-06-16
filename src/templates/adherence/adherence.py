import json
import os
from typing import List, Tuple

import datasets


_DESCRIPTION = """\
Whether a sentence from a research paper is about take-up/adherence/compliance or not."""

_HOMEPAGE = ""

_LICENSE = ""


class AdherenceDataset(datasets.GeneratorBasedBuilder):
    """Adherence Dataset"""

    data_dir = "src/templates/adherence"

    VERSION = datasets.Version("0.1.0")

    def _info(self):
        features = datasets.Features(
            {
                "sentence": datasets.Value("string"),
                "context": datasets.Value("string"),
                "label": datasets.Value("int8"),
                "section": datasets.Value("string"),
                "prediction": datasets.Value("float"),
                "intervention": datasets.Value("string"),
            }
        )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": datasets.Split.TRAIN},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"split": datasets.Split.VALIDATION},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": datasets.Split.TEST},
            ),
        ]

    def _generate_examples(self, split):
        with open(os.path.join(self.data_dir, "raw.json")) as f:
            data = json.load(f)

            rows: List[Tuple[int, dict]] = []

            for key, row in enumerate(data):
                label = (
                    1
                    if row["annotations"]
                    and row["annotations"][0].get("result")
                    and "adherence"
                    in row["annotations"][0]["result"][0]
                    .get("value", {})
                    .get("choices", [])
                    else 0
                )
                rows.append((key, {
                    "sentence": row["data"]["sentence"],
                    "context": row["data"]["context"],
                    "intervention": row["data"]["intervention"],
                    "section": row["data"]["section"],
                    "prediction": row["data"]["prediction"],
                    "label": label,
                }))

            if split == datasets.Split.TRAIN:
                rows = rows[:int(len(rows) * 0.9)]
            else:
                rows = rows[int(len(rows) * 0.9):]
            
            for row in rows:
                yield row
