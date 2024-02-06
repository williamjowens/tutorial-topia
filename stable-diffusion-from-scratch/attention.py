import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x: torch.Tensor, causal_mask=False):
        # x: [batch_size, seq_len, d_embed]
        input_shape = x.shape

        batch_size, seq_len, d_embed = input_shape

        interim_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # [batch_size, seq_len, d_embed] -> [batch_size, seq_len, d_embed * 3] -> 3 tensors of shape [batch_size, seq_len, d_embed]
        q, k, v = self.in_proj(x).chunk(3, dim=-1)

        # [batch_size, seq_len, d_embed] -> [batch_size, seq_len, H, d_embed / H] -> [batch_size, H, seq_len, d_embed / H]
        q = q.view(*interim_shape).transpose(1, 2)
        k = k.view(*interim_shape).transpose(1, 2)
        v = v.view(*interim_shape).transpose(1, 2)

        # [batch_size, H, seq_len, seq_len]
        weight = q @ k.transpose(-1, -2)

        #
        if causal_mask:
            # Mask where upper triangle above principal diagonal is comprised of 1s
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        # [batch_size, H, seq_len, seq_len] @ [batch_size, H, seq_len, d_embed / H] -> [batch_size, H, seq_len, d_embed / H]
        output = weight @ v

        # [batch_size, H, seq_len, d_embed / H] -> [batch_size, seq_len, H, d_embed / H]
        output = output.transpose(1, 2)

        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # [batch_size, seq_len, d_embed]
        return output


class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_embed, d_embed, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_embed // n_heads

    def forward(self, x, y):
        # x (latent): [batch_size, seq_len_Q, dim_Q]
        # y (context): [batch_size, seq_len_KV, dim_KV] = [batch_size, 77, 768]

        input_shape = x.shape
        batch_size, sequence_length, d_embed = input_shape

        interim_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply query by Wq
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)

        q = q.view(interim_shape).transpose(1, 2)
        k = k.view(interim_shape).transpose(1, 2)
        v = v.view(interim_shape).transpose(1, 2)

        weight = q @ k.transpose(-1, -2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output