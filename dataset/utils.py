# datasets:
# -- utils.py

import json
import io
import torch.distributed as dist
import jsonlines


# jsonl
def is_rank_0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode, encoding='utf8')
    return f


def jload(f, mode='r'):
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict


def jsonl_load(file):
    ret = []
    with jsonlines.open(file) as lines:
        for obj in lines:
            ret.append(obj)
    return ret
