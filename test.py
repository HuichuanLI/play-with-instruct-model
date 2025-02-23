import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.opt import OPTForCausalLM

if __name__ == '__main__':
    mn = 'facebook/opt-125m'
    cache = './cache/code/cache'
    tkr = AutoTokenizer.from_pretrained(mn, cache_dir=cache)
    model = AutoModelForCausalLM.from_pretrained(mn, cache_dir=cache)

    hello = ['how are you, are you ok?', 'i like china.']

    ids = tkr(hello, return_tensors='pt', padding='longest',
              max_length=128, truncation=True)
    print(ids)
    resp = tkr.decode(ids['input_ids'].numpy()[0])
    print(resp)
    # 12*[batch, seq_len, d_dim]
    # [batch, seq_len, n_vob]
    # resp = model(**ids)
    resp = model.generate(**ids).numpy()
    print(resp)
    a = tkr.decode(resp[0])
    print(a)
    b = tkr.decode(resp[1])
    print(b)
