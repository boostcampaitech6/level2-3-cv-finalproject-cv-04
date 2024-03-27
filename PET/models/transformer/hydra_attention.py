import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.nn.parameter import Parameter

Tensor = torch.Tensor

class HydraAttention(nn.Module):
    def __init__(self, embed_dim, num_heads ,kdim=None, vdim=None, dropout=0.0, bias=True, device=None, dtype=None):
        super(HydraAttention, self).__init__()
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.bias = bias

        self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)
        
        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))

        self.out_proj = nn.modules.linear.NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        

    def forward(
        self, 
        query: Tensor,
        key: Tensor,
        value: Tensor, 
        attn_mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None):
        # 512, 16 , 256 = T, B, D
        tgt_len, bsz, embed_dim = query.shape # 512, 16, 256
        src_len, _, _ = key.shape
        
        if key_padding_mask is not None:
            assert key_padding_mask.shape == (bsz, src_len), \
                f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
            key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
                expand(-1, 1, -1, -1).reshape(bsz, src_len, 1)
            if attn_mask is None:
                attn_mask = key_padding_mask
            else:
                attn_mask = attn_mask + key_padding_mask

        # Apply projections with bias to x for q, k, v
        q, k, v = nn.functional._in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)
        q = q.view(tgt_len, bsz, self.embed_dim).transpose(0, 1) # 128, 512 -> tgt_len, 32 -> head_dim
        k = k.view(k.shape[0], bsz, self.embed_dim).transpose(0, 1) # 4096, 512, 1 because head is 256
        v = v.view(v.shape[0], bsz, self.embed_dim).transpose(0, 1)

        # Normalize q and k
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        
        # Apply mask to k if provided
        if attn_mask is not None:
            # Ensure the mask is broadcastable to the shape of k
            k = k.masked_fill(attn_mask, 0)
        
        # Attention mechanism
        kvw = k * v
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2)
        out = q * kvw.sum(dim=-2, keepdim=True)
        # Reshape and apply output projection
        out = out.transpose(0, 1).contiguous().view(tgt_len * bsz , embed_dim)
        out = F.linear(out, self.out_proj.weight, self.out_proj.bias) # 8192, 256
        out = out.view(tgt_len, bsz, out.size(1)) # 512, 16, 256
        return out