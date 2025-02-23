# replay_buffer
# -- naive.py
import random
from typing import List
import torch
from exp_maker.base import Experience

from .base import ReplayBuffer
from .utils import BufferItem, make_experience_batch, split_experience_batch


class NaiveReplayBuffer(ReplayBuffer):
    def __init__(self, sample_batch_size: int,
                 limit: int = 0, cpu_offload: bool = True, device='cuda'):
        super(NaiveReplayBuffer, self).__init__(sample_batch_size, limit)
        self.cpu_offload = cpu_offload
        self.items: List[BufferItem] = []
        self.device = torch.device(device)

    @torch.no_grad()
    def append(self, experience: Experience) -> None:
        if self.cpu_offload:
            experience.to_device(torch.device('cpu'))
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        self.items.clear()

    @torch.no_grad()
    def sample(self) -> Experience:
        items = random.sample(self.items, self.sample_batch_size)
        experience = make_experience_batch(items)
        if self.cpu_offload:
            experience.to_device(self.device)
        return experience

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx) -> BufferItem:
        return self.items[idx]

    def collate_fn(self, batch) -> Experience:
        experience = make_experience_batch(batch)
        return experience
