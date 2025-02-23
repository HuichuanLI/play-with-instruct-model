# dataset
# -- rm_dataset.py

from typing import Optional, Callable
import torch
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm


class RmDataset(Dataset):
    def __init__(self, dataset, tokenizer: Callable,
                 max_len: int, special_token=None) -> None:
        super(RmDataset, self).__init__()
        # 相对更好的
        self.chosen = []
        # 相对更差的
        self.reject = []
        if special_token is None:
            self.end_token = tokenizer.eos_token
        else:
            self.end_token = special_token
        for data in tqdm(dataset):
            chosen = data['chosen'] + self.end_token
            chosen_token = tokenizer(chosen, max_length=max_len,
                                     padding='max_length', truncation=True, return_tensors='pt')
            self.chosen.append({'input_ids': chosen_token['input_ids'],
                                'attention_mask': chosen_token['attention_mask']})
            reject = data['rejected'] + self.end_token
            reject_token = tokenizer(reject, max_length=max_len,
                                     padding='max_length', truncation=True, return_tensors='pt')
            self.reject.append({'input_ids': reject_token['input_ids'],
                                'attention_mask': reject_token['attention_mask']})

    def __len__(self):
        return len(self.chosen)

    def __getitem__(self, idx):
        return (self.chosen[idx]['input_ids'], self.chosen[idx]['attention_mask'],
                self.reject[idx]['input_ids'], self.reject[idx]['attention_mask'])
