# trainer:
# -- base.py

from abc import ABC, abstractmethod
from typing import Any
import torch
import tqdm
from torch import nn
from torch.optim import Optimizer
from torch.utils._pytree import tree_map


def to_device(x: Any, device: torch.device):
    def _to(t: Any):
        if isinstance(t, torch.Tensor):
            return t.to(device)
        return t

    return tree_map(_to, x)


class Trainer(ABC):
    def __init__(self, max_epochs: int, model: nn.Module, optimizer: Optimizer):
        super(Trainer, self).__init__()
        self.max_epochs = max_epochs
        self.model = model
        self.optimizer = optimizer

    @abstractmethod
    def _train(self, epoch):
        raise NotImplementedError()

    @abstractmethod
    def _eval(self, epoch):
        raise NotImplementedError()

    def _before_fit(self):
        self.no_epoch_bar = False

    def fit(self, *args, **kwargs):
        self._before_fit(*args, **kwargs)
        for epoch in tqdm.trange(self.max_epochs, desc='epochs', disable=self.no_epoch_bar):
            self._train(epoch)
            self._eval(epoch)


from torch.utils.data import DataLoader


class CycledDataLoader:
    def __init__(self, dataloader: DataLoader):
        self.dataloader = dataloader
        self.count = 0
        self.dataloader_iter = iter(dataloader)

    def next(self):
        self.count += 1
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            self.count = 0
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)


class OnPolicyTrainer(ABC):
    def __init__(self, buffer: NaiveReplayBuffer):
        super(OnPolicyTrainer, self).__init__()
        self.buffer = buffer

    @abstractmethod
    def _make_experience(self, collect_step: int):
        raise NotImplementedError()

    @abstractmethod
    def _learn(self, update_step: int):
        raise NotImplementedError()

    def _collect_phase(self, collect_step: int):
        experience = self._make_experience(collect_step)
        self.buffer.append(experience)

    def _update_phase(self, update_step: int):
        self._learn(update_step)

    def fit(self, prompt_dataloader: DataLoader,
            pretrain_dataloader: DataLoader,
            num_episodes: int,
            num_collect_steps: int,
            num_update_steps: int
            ):
        self.prompt_dataloader = CycledDataLoader(prompt_dataloader)
        self.pretrain_dataloader = CycledDataLoader(pretrain_dataloader)
        for epis in tqdm.trange(num_episodes, desc='episodes'):
            for collect_step in tqdm.trange(num_collect_steps,
                                            desc='collect steps'):
                self._collect_phase(collect_step)
            for update_step in tqdm.trange(num_update_steps,
                                           desc='update steps'):
                self._update_phase(update_step)
            self.buffer.clear()
