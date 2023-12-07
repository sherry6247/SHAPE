'''
Author: your name
Date: 2022-05-02 15:50:25
LastEditTime: 2022-07-14 22:58:04
LastEditors: error: git config user.name && git config user.email & please set dead value or install git
'''
from collections import namedtuple
from einops import rearrange, repeat
import torch
from torch import einsum, nn
import torch.nn.functional as F
from torchtyping import patch_typeguard, TensorType
from typeguard import typechecked
from typing import Optional, Tuple
from x_transformers.x_transformers import (
    apply_rotary_pos_emb, default, exists, FeedForward, RMSNorm, RotaryEmbedding
)

patch_typeguard()

"""
This may change significantly as I work out how to implement this properly, but until then this is largely copied from Phil Wang (@lucidrains)
"""


SeqTensor = TensorType['batch', 'seq_len', 'token_dim']
StateTensor = TensorType['batch', 'state_len', 'state_dim']

# constants

DEFAULT_DIM_HEAD = 64
MIN_DIM_HEAD = 64 #32

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates'
])

def cast_tuple(val, num = 1):
    return val if isinstance(val, tuple) else ((val,) * num)

def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t: SeqTensor, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[-seq_len:, :]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


@typechecked
class RecurrentStateGate(nn.Module):
    """Poor man's LSTM
    """

    def __init__(self, dim: int):
        super().__init__()

        self.main_proj = nn.Linear(dim, dim, bias = True)
        self.input_proj = nn.Linear(dim, dim, bias = True)
        self.forget_proj = nn.Linear(dim, dim, bias = True)
    
    def forward(self, x: SeqTensor, state: StateTensor) -> StateTensor:
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)


class Attention(nn.Module):
    """Shamelessly copied from github.com/lucidrains/RETRO-pytorch
    """
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        null_kv = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        # allowing for attending to nothing (null function)
        # and to save attention from breaking if all retrieved chunks are padded out
        self.null_kv = nn.Parameter(torch.randn(2, inner_dim)) if null_kv else None

    def forward(self, x, mask = None, context = None, pos_emb = None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        x = self.norm(x)
        kv_input = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # scale
        q = q * scale

        # apply relative positional encoding (rotary embeddings)
        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num = 2)
            # print(f'q: {q.shape}\nq_pos_emb: {q_pos_emb.shape}')
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # add null key / values
        if exists(self.null_kv):
            nk, nv = self.null_kv.unbind(dim = 0)
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b = b, h = h), (nk, nv))
            k = torch.cat((nk, k), dim = -2)
            v = torch.cat((nv, v), dim = -2)

        # derive query key similarities
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking
        mask_value = -torch.finfo(sim.dtype).max

        
        if exists(mask):
            # 这里自己加上上三角矩阵，避免信息泄露
            tril_mask = torch.tril(torch.ones(mask.shape[-1], mask.shape[-1]).view(1,1,mask.shape[-1], mask.shape[-1])).bool().to(mask.device)
            sim = sim.masked_fill(~tril_mask, mask_value)
            if exists(self.null_kv):
                mask = F.pad(mask, (1, 0), value = True)

            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention
        attn = sim.softmax(dim = -1)

        attn = self.dropout(attn)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # combine heads linear out
        return self.to_out(out), None


@typechecked
class BlockRecurrentAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        dim_state: int,
        dim_head: int = DEFAULT_DIM_HEAD,
        state_len: int = 512,
        heads: int = 8,
        **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        attn_kwargs = {}

        self.dim = dim
        self.dim_state = dim_state

        self.heads = heads
        self.causal = True
        self.state_len = state_len
        rotary_emb_dim = max(dim_head // 2, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)
        
        self.input_self_attn = Attention(dim, heads = heads, causal = True, **attn_kwargs)
        self.state_self_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)

        self.input_state_cross_attn = Attention(dim, heads = heads, causal = False, **attn_kwargs)
        self.state_input_cross_attn = Attention(dim_state, heads = heads, causal = False, **attn_kwargs)

        self.proj_gate = RecurrentStateGate(dim)
        self.ff_gate = RecurrentStateGate(dim)

        self.input_proj = nn.Linear(dim + dim_state, dim, bias = False)
        self.state_proj = nn.Linear(dim + dim_state, dim, bias = False)

        self.input_ff = FeedForward(dim)
        self.state_ff = FeedForward(dim_state)

    def forward(
        self,
        x: SeqTensor,
        state: Optional[StateTensor] = None,
        mask = None,
        state_mask = None,
        rel_pos = None,
        rotary_pos_emb = None,
        prev_attn = None,
        mem = None
    ) -> Tuple[SeqTensor, StateTensor]:
        batch, seq_len, device = x.shape[0], x.shape[-2], x.device
        if not exists(state):
            state = torch.zeros((batch, self.state_len, self.dim_state)).to(x.device)
        self_attn_pos_emb = self.rotary_pos_emb(seq_len, device = device)
        state_pos_emb = self.rotary_pos_emb(self.state_len, device = device)
        input_attn, _ = self.input_self_attn(x, mask = mask, pos_emb = self_attn_pos_emb)
        state_attn, _ = self.state_self_attn(state, mask = state_mask, pos_emb = state_pos_emb)

        # This actually is different from how it is implemented in the paper, because the Keys and Values aren't shared
        # between the cross attention and self-attention. I'll implement that later, this is faster for now.
        input_as_q_cross_attn, _ = self.input_state_cross_attn(x, context = state, mask = mask) # , context_mask = state_mask # [batch, seq_len, hd]
        state_as_q_cross_attn, _ = self.state_input_cross_attn(state, context = x, mask = state_mask) # , context_mask = mask  # [batch, state_len, hd]

        projected_input = self.input_proj(torch.cat((input_as_q_cross_attn, input_attn), dim=2)) # torch.concat
        projected_state = self.state_proj(torch.cat((state_as_q_cross_attn, state_attn), dim=2)) # torch.concat

        input_residual = projected_input + x
        state_residual = self.proj_gate(projected_state, state)

        output = self.input_ff(input_residual) + input_residual
        next_state = self.ff_gate(self.state_ff(state_residual), state_residual)

        return output, next_state