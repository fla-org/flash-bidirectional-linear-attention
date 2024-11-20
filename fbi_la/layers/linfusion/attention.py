import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/public/liguoqi/ssl/wds/flash-non-causal-linear-attention')))
from fla_nc.ops.linear_attn.attention import linear_attention

from einops import rearrange


def get_none_linear_projection(query_dim, mid_dim=None):
    # If mid_dim is None, then the mid_dim is the same as query_dim
    # If mid_dim is -1, then no non-linear projection is used, and the identity is returned
    return (
        torch.nn.Sequential(
            torch.nn.Linear(query_dim, mid_dim or query_dim),
            torch.nn.LayerNorm(mid_dim or query_dim),
            torch.nn.LeakyReLU(inplace=True),
            torch.nn.Linear(mid_dim or query_dim, query_dim),
        )
        if mid_dim != -1
        else torch.nn.Identity()
    )


class GeneralizedLinearAttention(Attention):
    def __init__(self, *args, projection_mid_dim=None, **kwargs):
        """
        Args:
            query_dim: the dimension of the query.
            out_dim: the dimension of the output.
            dim_head: the dimension of the head. (dim_head * num_heads = query_dim)
            projection_mid_dim: the dimension of the intermediate layer in the non-linear projection. 
              If `None`, then the dimension is the same as the query dimension.
              If `-1`, then no non-linear projection is used, and the identity is returned.
        """
        super().__init__(*args, **kwargs)
        self.add_non_linear_model(projection_mid_dim)

    def from_attention_instance(attention_instance, projection_mid_dim=None):
        assert isinstance(attention_instance, Attention)
        new_instance = GeneralizedLinearAttention(128)
        new_instance.__dict__ = attention_instance.__dict__
        new_instance.add_non_linear_model(mid_dim = projection_mid_dim)
        return new_instance

    def add_non_linear_model(self, mid_dim=None, **kwargs):
        query_dim = self.to_q.weight.shape[0]
        self.to_q_ = get_none_linear_projection(query_dim, mid_dim, **kwargs)
        self.to_k_ = get_none_linear_projection(query_dim, mid_dim, **kwargs)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        mode='triton',
        **kwargs,
    ):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        _, sequence_length, _ = hidden_states.shape

        query = self.to_q(hidden_states + self.to_q_(hidden_states))
        key = self.to_k(encoder_hidden_states + self.to_k_(encoder_hidden_states))
        value = self.to_v(encoder_hidden_states)

        query = self.head_to_batch_dim(query)
        key = self.head_to_batch_dim(key)
        value = self.head_to_batch_dim(value)
        
        query = F.elu(query) + 1.0
        key = F.elu(key) + 1.0

        if mode == 'torch':
            z = query @ key.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-4
            kv = (key.transpose(-2, -1) * (sequence_length**-0.5)) @ (
                value * (sequence_length**-0.5)
            )
            hidden_states = query @ kv / z
        elif mode == 'triton':
            query, key, value = map(lambda x: rearrange(x, '(b h) l d -> b h l d', h=self.heads), [query, key, value])

            hidden_states = linear_attention(query, key, value, 1.0)
            
            hidden_states = rearrange(hidden_states, 'b h l d -> (b h) l d')
        else:
            raise NotImplementedError

        hidden_states = self.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)
        # dropout
        hidden_states = self.to_out[1](hidden_states)

        return hidden_states
    
    
if __name__ == "__main__":
    B, H, L, D = 4, 16, 512, 64
    dtype = torch.float32

    x = torch.randn((B, L, H*D), dtype=dtype, device="cuda", requires_grad=True)
    do = torch.randn_like(x).cuda()
    
    model = GeneralizedLinearAttention(query_dim=H*D, out_dim=H*D, dim_head=D).cuda()
    
    # naive
    ref = model(x, mode='torch')
    
    x.retain_grad()
    ref.backward(do, retain_graph=True)
    ref_dx, x.grad = x.grad.clone(), None
    
    # triton
    tri = model(x, mode='triton')
    
    x.retain_grad()
    tri.backward(do, retain_graph=True)
    tri_dx, x.grad = x.grad.clone(), None
    
    assert torch.allclose(ref, tri, rtol=0, atol=1e-4)
    assert torch.allclose(ref_dx, tri_dx, rtol=0, atol=1e-4)
    print("Pass")