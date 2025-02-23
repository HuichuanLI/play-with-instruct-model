# dataset
# -- prompt_dataset.py
from collections import defaultdict
from typing import Dict
import torch
import transformers
from torch.utils.data import Dataset
import logging
from .utils import jsonl_load, jload

logger = logging


class PromptDataset(Dataset):
    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 max_len: int = 96,
                 device='cuda'
                 ):
        super(PromptDataset, self).__init__()
        self.keyed_prompt = defaultdict(list)
        logger.info('loading data ...')
        list_data_dict = jsonl_load(data_path)
        logger.info(f'loaded {len(list_data_dict)} examples.')
        instructions = [data_dict['instruction'] for data_dict in list_data_dict]
        tokens = tokenizer(instructions,
                           return_tensors='pt',
                           max_length=max_len,
                           padding='max_length',
                           truncation=True
                           )
        # [len of list, max_len]
        # list[max_len]
        for k, tensor in tokens.items():
            self.keyed_prompt[k] = tensor.to(device).unbind()

    def __len__(self):
        return len(self.keyed_prompt['input_ids'])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.keyed_prompt.items()}
