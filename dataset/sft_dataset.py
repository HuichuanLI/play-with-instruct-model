# datasets:
# -- sft_dataset.py
# LM abcde, SFT-response
import copy
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence
import torch
# huggingface
import transformers
from torch.utils.data import Dataset
from tqdm import tqdm

from .utils import jsonl_load
import logging

logger = logging

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input":
        ("Below is an instruction that describes a task, paired with an input that provides further context. "
         "Write a response that appropriately completes the request.\n\n"
         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"),
    "prompt_no_input": ("Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{instruction}\n\n### Response:"),
}


def _tokenize_fn(strings: List[str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_len: int) -> Dict[str, torch.Tensor]:
    tokenized_list = tokenizer(strings, return_tensors='pt',
                               padding='longest', max_length=max_len, truncation=True)

    input_ids = labels = tokenized_list['input_ids']
    input_ids_lens = labels_lens = tokenized_list['input_ids'].ne(tokenizer.pad_token_id).sum(dim=-1)
    return dict(input_ids=input_ids,
                labels=labels,
                input_ids_lens=input_ids_lens,
                labels_lens=labels_lens
                )


def preprocess(sources: Sequence[str], targets: Sequence[str],
               tokenizer: transformers.PreTrainedTokenizer, max_len: int) -> Dict:
    examples = [s + t for s, t in zip(sources, targets)]
    examples_token, source_token = [
        _tokenize_fn(s, tokenizer, max_len)
        for s in (examples, sources)
    ]
    input_ids = examples_token['input_ids']
    labels = copy.deepcopy(input_ids)
    for lab, src_len in zip(labels, source_token['input_ids_lens']):
        lab[:src_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


class SupervisedDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_len: int = 512):
        super(SupervisedDataset, self).__init__()
        logger.info('loading data ...')
        list_data_dict = jsonl_load(data_path)
        logger.info(f'load {len(list_data_dict)} examples')
        logger.info('formatting inputs...')
        prompt_input, prompt_no_input = PROMPT_DICT['prompt_input'], PROMPT_DICT['prompt_no_input']
        sources = [prompt_input.format_map(example)
                   if example.get('input') is not None else prompt_no_input.format_map(example)
                   for example in list_data_dict
                   ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]
        logger.info('tokenizing inputs...')
        data_dict = preprocess(sources, targets, tokenizer, max_len)
        self.input_ids = data_dict['input_ids']
        self.labels = data_dict['labels']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return dict(input_ids=self.input_ids[item], labels=self.labels[item])


@dataclass
class CollatorForSupervisedDataset(object):
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([ins[key] for ins in instances]
                                  for key in ('input_ids', 'labels'))
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True,
                                                    padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True, padding_value=-100)
        return dict(input_ids=input_ids, labels=labels,
                    attention_mask=input_ids.ne(self.tokenizer.pad_token_id))


from transformers import AutoTokenizer

if __name__ == '__main__':
    tokenr = AutoTokenizer.from_pretrained('facebook/opt-125m')
    s = ["哈哈哈","我爱你"]
    ret = tokenr(s)
    print(ret)

    # a = '''"Below is an instruction that describes a task, paired with an input that provides further context. "
    # "Write a response that appropriately completes the request.\n\n"
    #      "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"'''
    # b = a.format_map({'instruction': '123', 'input': '456'})
    # print(b)
