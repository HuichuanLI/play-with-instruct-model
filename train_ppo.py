# train_ppo.py

import argparse
import torch
from dataset.prompt_dataset import PromptDataset
from dataset.sft_dataset import SupervisedDataset, CollatorForSupervisedDataset

from model.opt import OPTRM, OPTActor, OPTCritic
from trainer.ppo import PPOTrainer
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM


# from
def main(args):
    initial_model = AutoModelForCausalLM. \
        from_pretrained(args.pretrain, cache_dir=args.cache)
    actor = OPTActor(pretrained=args.pretrain, cache_dir=args.cache)
    if args.actor_path is not None:
        state_dict = torch.load(args.actor_path,
                                map_location='cpu')
        actor.load_state_dict(state_dict)
        del state_dict

    reward_model = OPTRM(pretrained=args.rm_pretrain, cache_dir=args.rm_cache)
    critic = OPTCritic(pretrained=args.rm_pretrain,
                       cache_dir=args.rm_cache, use_action_mask=True)
    if args.critic_path is not None:
        state_dict = torch.load(args.critic_path,
                                map_location='cpu')
        critic.load_state_dict(state_dict)
        del state_dict

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain,
                                              cache_dir=args.cache)
    tokenizer.pad_token = tokenizer.eos_token

    actor_optim = Adam(actor.parameters(), lr=1e-7)
    critic_optim = Adam(critic.parameters(), lr=1e-7)

    data_collator = CollatorForSupervisedDataset(tokenizer)

    prompt_dataset = PromptDataset(tokenizer=tokenizer,
                                   data_path=args.prompt_dataset,
                                   device=args.device)
    prompt_dataloader = DataLoader(prompt_dataset,
                                   shuffle=True, batch_size=args.exp_batch_size)

    pretrain_dataset = SupervisedDataset(tokenizer=tokenizer,
                                         data_path=args.pretrain_dataset,
                                         max_len=args.max_input_len)
    pretrain_dataloader = DataLoader(pretrain_dataset,
                                     shuffle=True, batch_size=args.ptx_batch_size,
                                     collate_fn=data_collator)

    trainer = PPOTrainer(actor, critic, reward_model,
                         initial_model, actor_optim, critic_optim,
                         kl_coef=args.kl_coef,
                         ptx_coef=args.ptx_coef,
                         train_batch_size=args.train_batch_size,
                         max_length=args.max_seq_len,
                         # [a,b,c,d][x,y,z]-> abcd->x, abcdx->y, abcdxy->z.
                         # abcd->x, (abcd)x->y, y->z
                         use_cache=True,
                         do_sample=True,
                         temperature=1.0,
                         top_k=50,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id
                         )
    trainer.fit(prompt_dataloader=prompt_dataloader,
                pretrain_dataloader=pretrain_dataloader,
                num_episodes=args.num_episodes,
                num_collect_steps=args.num_collect_steps,
                num_update_steps=args.num_update_steps
                )

    state_dict = actor.state_dict()
    torch.save(state_dict, args.actor_path)
    state_dict = critic.state_dict()
    torch.save(state_dict, args.critic_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
                        default='cpu')
    parser.add_argument('--cache', type=str,
                        default='cache/code/cache')
    parser.add_argument('--rm_cache', type=str,
                        default='cache/code/cache')
    parser.add_argument('--prompt_dataset',
                        type=str, default='ds/test2.json', help='path to the prompt dataset')
    parser.add_argument('--pretrain_dataset',
                        type=str, default='ds/test.jsonl', help='path to the pretrained dataset')
    parser.add_argument('--pretrain', type=str, default='facebook/opt-125m')
    parser.add_argument('--actor_path', type=str, default=None)
    parser.add_argument('--critic_path', type=str, default=None)
    parser.add_argument('--rm_pretrain', type=str, default='facebook/opt-125m')
    # parser.add_argument('--save_path', type=str, default='actor_checkpoint_prompts')
    # parser.add_argument('--need_optim_ckpt', type=bool, default=False)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--num_collect_steps', type=int, default=10)
    parser.add_argument('--num_update_steps', type=int, default=5)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--ptx_batch_size', type=int, default=1)
    parser.add_argument('--exp_batch_size', type=int, default=8)
    # parser.add_argument('--lora_rank', type=int, default=0, help="low-rank adaptation matrices rank")
    parser.add_argument('--kl_coef', type=float, default=0.1)
    parser.add_argument('--ptx_coef', type=float, default=0.9)
    parser.add_argument('--max_input_len', type=int, default=96)
    parser.add_argument('--max_seq_len', type=int, default=128)
    args = parser.parse_args()
    main(args)
