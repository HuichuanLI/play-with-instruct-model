# root
# train_rm.py
import argparse
import torch
from dataset.rm_dataset import RmDataset
from model.loss import LogSigLoss
from model.opt.opt_rm import OPTRM
from trainer.rm import RewardModelTrainer
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from dataset.utils import jsonl_load


def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain, cache_dir=args.cache)
    tokenizer.pad_token = tokenizer.eos_token

    device = 'cpu'
    model = OPTRM(pretrained=args.pretrain, cache_dir=args.cache).to(device)
    if args.model_path is not None:
        state_dict = torch.load(args.model_path)
        model.load_state_dict(state_dict)

    optim = Adam(model.parameters(), lr=5e-6)
    loss_fn = LogSigLoss()
    train_data = jsonl_load(args.dataset)
    train_dataset = RmDataset(train_data, tokenizer, args.max_len)
    valid_dataset = RmDataset(train_data, tokenizer, args.max_len)
    eval_dataset = RmDataset(train_data, tokenizer, args.max_len)

    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                  batch_size=args.batch_size, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, shuffle=True,
                                  batch_size=args.batch_size, pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True,
                                 batch_size=args.batch_size, pin_memory=True)

    lr_scheduler = CosineAnnealingLR(optim, len(train_dataloader) // 100)
    trainer = RewardModelTrainer(model=model,
                                 optim=optim, lr_scheduler=lr_scheduler,
                                 loss_fn=loss_fn, max_epochs=args.max_epochs, device=device
                                 )

    trainer.fit(train_dataloader=train_dataloader,
                valid_dataloader=valid_dataloader,
                eval_dataloader=eval_dataloader
                )

    state_dict = model.state_dict()
    torch.save(state_dict, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, choices=['Anthropic/hh-rlhf', 'Dahoas/rm-static'], default='ds/rm.jsonl')
    parser.add_argument('--cache', type=str, default='./cache/code/cache')
    parser.add_argument('--save_path', type=str, default='rm_ckpt')
    parser.add_argument('--max_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=512)
    # parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--loss_fn', type=str, default='log_sig')
    parser.add_argument('--test', type=bool, default=False)
    args = parser.parse_args()
    train(args)
