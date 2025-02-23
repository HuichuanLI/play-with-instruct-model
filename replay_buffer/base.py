# replay_buffer
# -- base.py

from abc import ABC, abstractmethod
from typing import Any
from exp_maker.base import Experience


class ReplayBuffer(ABC):
    def __init__(self, sample_batch_size: int, limit: int = 0):
        super(ReplayBuffer, self).__init__()
        self.sample_batch_size = sample_batch_size
        self.limit = limit

    @abstractmethod
    def append(self, exp: Experience) -> None:
        pass

    @abstractmethod
    def clear(self) -> None:
        pass

    @abstractmethod
    def sample(self) -> Experience:
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, itx):
        pass

    @abstractmethod
    def collate_fn(self, batch: Any) -> Experience:
        pass
