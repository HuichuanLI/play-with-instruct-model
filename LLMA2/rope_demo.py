import torch


# f(q,m) = q*e^I*m*theta = q*e^im10000^(-2i/d)    i->(0,2/d -1)

# dim =2 end = 3

def precompute_freqs_cis(dim, end, theta=10000):
    # e^i*m*(10000)^-2i/d ->(10000)^(-2i/d) i->(0,2/d -1)
    # 1/(10000**[0.0]/2)     shape [1]
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
    # [3]
    m = torch.arange(end)
    # [3],[1] ->[3,1]
    freqs = torch.outer(m, freqs)

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

    return freqs_cis


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # [3,1] ->[1,3,1,1]
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq, xk, freqs_cis):
    # 复数 实部+虚部   xq shape [1,3,2] [1,3,1,2]   *xq.shape[:-1]  [1,3,1,2] -> view_as_complex ->[1,3,1,1]
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))  # [1,3,1,1]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))

    # freqs_cis [3,1] ->[1,3,1,1]
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    # q*e^I*m*theta
    # [1,3,2]  xq_*ffreqs_cis->[1,3,1,1]->view_as_real->[1,3,1,2]->[1,3,2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    # xq [1,3,1,2]->[1,3,1,1,2]->VIEW_CIMPLEX-》[1,3,1,1,1]->view_as_real->[1,3,1,1,2]->[1,3,1,2]
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xq)

# dim = 2
# end = 3
#
# fres_cis = precompute_ferqs_cis(dim,end)
# xq = torch.randn(1,end,dim)
# xk = torch.randn(1,end,dim)
#
# apply_rotary_emb(xq,xk,fres_cis)
