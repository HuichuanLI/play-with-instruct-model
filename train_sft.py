# root:
# -- train_sft.py
import argparse
from dataclasses import dataclass
import torch
import transformers
from typing import Callable, Dict, List, Sequence
import math

from dataset.sft_dataset import SupervisedDataset, CollatorForSupervisedDataset
from trainer.sft import SFTTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers.trainer import get_scheduler
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from peft import LoraConfig, get_peft_model, PeftType, PromptTuningConfig, TaskType, PrefixTuningConfig
from dataset.sft_dataset import SupervisedDataset
from trainer.sft import SFTTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers.trainer import get_scheduler
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


def train(args):
    # chatGLM
    # ~/.cache
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain,
                                              trust_remote_code=True, cache_dir=args.cache)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.pretrain,
                                                 trust_remote_code=True, cache_dir=args.cache)
    #     config = LoraConfig(r=4,
    #                         lora_alpha=4,
    #                         enable_lora=[True, False, True],
    #                         lora_dropout=0.1,
    #                         bias="none", )
    # config = PromptTuningConfig
    # model = get_peft_model(model, config)
    # config = PromptTuningConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     num_virtual_tokens=10,
    #     peft_type="PROMPT_TUNING",
    #     encoder_reparameterization_type="LSTM"
    # )

    # model = get_peft_model(model, config)
    # model.print_trainable_parameters()
    config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM,
                                num_virtual_tokens=10)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    optim = Adam(model.parameters(), lr=args.lr)
    train_dataset = SupervisedDataset(tokenizer=tokenizer,
                                      data_path=args.dataset,
                                      max_len=args.max_len
                                      )
    eval_dataset = None
    data_collator = CollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size,
                                  collate_fn=data_collator, pin_memory=True)
    # gpu cpu temp mem(pin mem)
    eval_dataloader = None
    step_per_epoch = len(train_dataloader) // args.batch_size
    max_steps = math.ceil(args.max_epochs * step_per_epoch)
    lr_scheduler = get_scheduler('cosine', optim,
                                 num_warmup_steps=math.ceil(max_steps * 0.03),
                                 num_training_steps=max_steps
                                 )
    device = 'cpu'
    trainer = SFTTrainer(model=model, optim=optim,
                         lr_scheduler=lr_scheduler, max_epochs=args.max_epochs, device=device
                         )
    trainer.fit(train_dataloader=train_dataloader, eval_dataloader=eval_dataloader,
                logger=logging
                )

    state = model.state_dict()
    torch.save(state, args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--dataset', type=str, default='ds/alpaca-en.json')
    parser.add_argument('--cache', type=str, default='cache/code/cache')
    parser.add_argument('--save_path', type=str, default='output')
    parser.add_argument('--max_epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--lr', type=float, default=5e-6)
    args = parser.parse_args()
    train(args)
