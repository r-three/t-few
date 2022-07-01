from typing import List
import torch
import numpy as np
from pytorch_lightning import LightningDataModule
from promptsource.templates import Template
from torch.utils.data import WeightedRandomSampler

from torch.utils.data import Dataset, Sampler, DistributedSampler
from typing import Union, Iterable, Sized, Optional, List, Any, Iterator
from pytorch_lightning.utilities.exceptions import MisconfigurationException


class _DatasetSamplerWrapper(Dataset):
    """Dataset to create indexes from `Sampler` or `Iterable`"""

    def __init__(self, sampler: Union[Sampler, Iterable]) -> None:
        if not isinstance(sampler, Sized):
            raise MisconfigurationException(
                "You seem to have configured a sampler in your DataLoader which"
                " does not provide `__len__` method. The sampler was about to be"
                " replaced by `DistributedSamplerWrapper` since `replace_sampler_ddp`"
                " is True and you are using distributed training. Either provide `__len__`"
                " method in your sampler, remove it from DataLoader or set `replace_sampler_ddp=False`"
                " if you want to handle distributed sampling yourself."
            )
        if len(sampler) == float("inf"):
            raise MisconfigurationException(
                "You seem to have configured a sampler in your DataLoader which"
                " does not provide finite `__len__` method. The sampler was about to be"
                " replaced by `DistributedSamplerWrapper` since `replace_sampler_ddp`"
                " is True and you are using distributed training. Either provide `__len__`"
                " method in your sampler which returns a finite number, remove it from DataLoader"
                " or set `replace_sampler_ddp=False` if you want to handle distributed sampling yourself."
            )
        self._sampler = sampler
        # defer materializing an iterator until it is necessary
        self._sampler_list: Optional[List[Any]] = None

    def __getitem__(self, index: int) -> Any:
        if self._sampler_list is None:
            self._sampler_list = list(self._sampler)
        return self._sampler_list[index]

    def __len__(self) -> int:
        return len(self._sampler)

    def reset(self) -> None:
        """Reset the sampler list in order to get new sampling."""
        self._sampler_list = list(self._sampler)


class DistributedSamplerWrapper(DistributedSampler):
    """Wrapper over ``Sampler`` for distributed training.
    Allows you to use any sampler in distributed mode. It will be automatically used by PyTorch Lightning in distributed
    mode if `replace_sampler_ddp=True`
    """

    def __init__(self, sampler: Union[Sampler, Iterable], *args: Any, **kwargs: Any) -> None:
        super().__init__(_DatasetSamplerWrapper(sampler), *args, **kwargs)

    def __iter__(self) -> Iterator:
        self.dataset.reset()
        return (self.dataset[index] for index in super().__iter__())


class FinetuneDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        if self.config.few_shot:
            _ = self.dataset_reader.read_few_shot_dataset()

    def setup(self, stage):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        if self.config.few_shot:
            self.train_dataset = self.dataset_reader.read_few_shot_dataset()
        else:
            self.train_dataset = self.dataset_reader.read_orig_dataset("train")
        self.dev_dataset = self.dataset_reader.read_orig_dataset("validation")
        self.train_dataset = FinetuneDatasetWithTemplate(
            self.train_dataset, self.dataset_reader.get_train_template(), self.tokenizer
        )
        self.dev_dataset = FinetuneDatasetWithTemplate(
            self.dev_dataset, self.dataset_reader.get_eval_template(), self.tokenizer
        )
        print(f"Train size {len(self.train_dataset)}")
        print(f"Eval size {len(self.dev_dataset)}")

    def get_balanced_sampler(self, dataset: "FinetuneDatasetWithTemplate"):
        targets = np.array([dataset[i][3].item() for i in range(len(dataset))])
        class_sample_count = np.array([len(np.where(targets == t)[0]) for t in np.unique(targets)])
        weight = 1.0 / class_sample_count
        samples_weight = np.array([weight[t] for t in targets])
        samples_weight = torch.from_numpy(samples_weight).float()
        return DistributedSamplerWrapper(WeightedRandomSampler(samples_weight, len(samples_weight)))

    def train_dataloader(self):
        kwargs = dict(sampler=self.get_balanced_sampler(self.train_dataset) if self.config.balanced_sampling else dict())
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False if self.config.balanced_sampling else True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
            **kwargs
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dev_dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=False),
            num_workers=min([self.config.eval_batch_size, self.config.num_workers]),
        )


class FinetuneDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, templates: List[Template], tokenizer, add_special_tokens=True):
        super().__init__()
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(self.templates, list):
            template: Template = np.random.choice(self.templates)
        else:
            template = self.templates
        example = self.dataset[key]
        input_str, target_str = template.apply(example)

        answer_choices = template.get_answer_choices_list(example)
        if isinstance(input_str, list):
            input_ids = torch.cat(
                [
                    self.tokenizer(
                        input_field, return_tensors="pt", truncation=True, add_special_tokens=False
                    ).input_ids.squeeze(0)
                    for input_field in input_str[:-1]
                ]
                + [
                    self.tokenizer(
                        input_str[-1], return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
                    ).input_ids.squeeze(0)
                ]
            )
        else:
            input_ids = self.tokenizer(
                input_str, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
            ).input_ids.squeeze(0)
        target_ids = self.tokenizer(
            target_str, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
        ).input_ids.squeeze(0)
        answer_choices_ids = [
            self.tokenizer(
                answer_choice, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
            ).input_ids.squeeze(0)
            for answer_choice in answer_choices
        ]
        label = torch.LongTensor([example["label"]])
        idx = torch.LongTensor([example["idx"]])
        return input_ids, target_ids, answer_choices_ids, label, idx


class PretrainDataModule(LightningDataModule):
    def __init__(self, config, tokenizer, dataset_reader):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.dataset_reader = dataset_reader

    def setup(self, stage):
        self.train_datasets = self.dataset_reader.read_orig_dataset("train")
        self.base_templates = self.dataset_reader.get_template()
        self.train_datasets_withtemplate = []
        for index, train_dataset in enumerate(self.train_datasets):
            self.train_datasets_withtemplate.append(
                PretrainDatasetWithTemplate(train_dataset, self.base_templates[index], self.tokenizer)
            )

        self.train_dataset = torch.utils.data.ConcatDataset(self.train_datasets_withtemplate)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=create_collate_fn(self.tokenizer.pad_token_id, pretrain=True),
            drop_last=True,
            num_workers=min([self.config.batch_size, self.config.num_workers]),
        )


class PretrainDatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, templates, tokenizer):
        super().__init__()
        self.dataset = dataset
        self.templates = templates
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(self.templates, list):
            template = np.random.choice(self.templates)
        else:
            template = self.templates
        example = self.dataset[key]
        input_target_str = template.apply(example)
        if len(input_target_str) == 2:
            input_str, target_str = input_target_str
            if target_str == "":
                target_str = "<NO LABEL>"
        else:
            input_str = "<NO INPUT>"
            target_str = "<NO LABEL>"
        input_ids = self.tokenizer(input_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        target_ids = self.tokenizer(target_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        return input_ids, target_ids


def create_collate_fn(pad_token_id, pretrain):
    def collate_fn(batch):
        if not pretrain:
            input_ids, target_ids, answer_choices_ids, labels, idx = zip(*batch)
        else:
            input_ids, target_ids = zip(*batch)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=pad_token_id)
        output_batch = {
            "input_ids": input_ids,
            "target_ids": target_ids,
        }

        if not pretrain:
            flat_answer_choice_ids = [choice for list_choices in answer_choices_ids for choice in list_choices]
            num_choice = [len(list_choices) for list_choices in answer_choices_ids]
            if max(num_choice) != min(num_choice):
                raise NotImplementedError("The collate_fn is not implmented for variable number of choices")
            flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
                flat_answer_choice_ids, batch_first=True, padding_value=pad_token_id
            )
            answer_choices_ids = flat_answer_choices_ids.view(len(answer_choices_ids), max(num_choice), -1).contiguous()
            labels = torch.cat(labels)
            idx = torch.cat(idx)
            output_batch.update(
                {
                    "answer_choices_ids": answer_choices_ids,
                    "labels": labels,
                    "idx": idx,
                }
            )

        return output_batch

    return collate_fn
