from datetime import datetime
from typing import Callable
import pandas as pd
import torch
# from tqdm import tqdm
import tqdm
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from .base import Trainer


class RewardModelTrainer(Trainer):
    def __init__(self, model,
                 optim: Optimizer,
                 lr_scheduler,
                 loss_fn: Callable,
                 max_epochs: int = 1,
                 device='cuda') -> None:
        super(RewardModelTrainer, self).__init__(max_epochs, model, optim)
        self.loss_fn = loss_fn
        self.scheduler = lr_scheduler
        self.device = torch.device(device)

    def _before_fit(self,
                    train_dataloader: DataLoader,
                    valid_dataloader: DataLoader,
                    eval_dataloader: DataLoader
                    ):
        super()._before_fit()
        self.datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.eval_dataloader = eval_dataloader

    def _eval(self, epoch):
        if self.eval_dataloader is None:
            return
        dist, on, cnt = 0, 0, 0
        with torch.no_grad():
            for chosen_ids, c_mask, reject_ids, r_mask in self.eval_dataloader:
                chosen_ids = chosen_ids.squeeze(1).to(self.device)
                # [batch, 1, seq_len]
                c_mask = c_mask.squeeze(1).to(self.device)
                # [batch, seq_len]
                chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
                reject_ids = reject_ids.squeeze(1).to(self.device)
                # [batch, 1, seq_len]
                r_mask = r_mask.squeeze(1).to(self.device)
                # [batch, seq_len]
                reject_reward = self.model(reject_ids, attention_mask=r_mask)
                for i in range(len(chosen_reward)):
                    cnt += 1
                    if chosen_reward[i] > reject_reward[i]:
                        on += 1
                dist += (chosen_reward - reject_reward).mean().item()
            self.dist = dist / len(self.eval_dataloader)
            self.acc = on / cnt

        log = pd.DataFrame(
            [[(epoch + 1) * len(self.train_dataloader),
              self.loss.item(), self.dist, self.acc]],
            columns=['step', 'loss', 'dist', 'acc']
        )
        log.to_csv('log.csv', mode='a', header=False, index=False)

    def _train(self, epoch):
        self.model.train()
        # 先自定义个进度条
        step_bar = tqdm.trange(len(self.train_dataloader),
                               desc='train step of %d' % epoch)
        cnt = 0
        for chosen_ids, c_mask, reject_ids, r_mask in self.train_dataloader:
            # 和eval 一样的
            chosen_ids = chosen_ids.squeeze(1).to(self.device)
            # [batch, 1, seq_len]
            c_mask = c_mask.squeeze(1).to(self.device)
            # [batch, seq_len]
            chosen_reward = self.model(chosen_ids, attention_mask=c_mask)
            reject_ids = reject_ids.squeeze(1).to(self.device)
            # [batch, 1, seq_len]
            r_mask = r_mask.squeeze(1).to(self.device)
            # [batch, seq_len]
            reject_reward = self.model(reject_ids, attention_mask=r_mask)
            self.loss = self.loss_fn(chosen_reward, reject_reward)
            self.loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            cnt += 1
            if cnt % 100 == 0:
                self.scheduler.step()
            step_bar.update()
        step_bar.close()
