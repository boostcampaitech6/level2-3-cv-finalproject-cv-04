"""
Transformer Encoder and Encoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .utils import *


class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=2, 
                 dim_feedforward=512, dropout=0.0,
                 activation="relu", **kwargs):
        super().__init__()
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, **kwargs)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)
        self.num_layers = num_encoder_layers
        self.norm = nn.LayerNorm(d_model)
        self._reset_parameters()
        self.enc_win_list = kwargs["encoder_window_list"]


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    
    def forward(self, src, src_pos, **kwargs):
        B, C, H, W = src.shape
        
        output = src  # [B C H W]
        for idx, (layer, enc_win_size) in enumerate(zip(self.layers, self.enc_win_list)):
            # encoder window partition
            src_win = window_partition(output, enc_win_size[1], enc_win_size[0], "batch_first")  # [L B' C]
            src_pos_win = window_partition(src_pos, enc_win_size[1], enc_win_size[0], "batch_first")  # [L B' C]

            # encoder forward
            output = layer(src_win, src_pos_win, **kwargs)  # [L B' C]

            if idx+1 == self.num_layers and self.norm is not None:
                output = self.norm(output)
            
            # reverse encoder window
            output = window_reverse(output, enc_win_size[1], enc_win_size[0], H, W, "batch_first")  # [B C H W]
            
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu", **kwargs):
        super().__init__()
        self.self_attn = WindowAttention(d_model, nhead, attn_drop=dropout, proj_drop=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


    def forward(self, src, src_pos: Optional[Tensor] = None, **kwargs):
        src = self.with_pos_embed(src, src_pos)
        
        # encoder self attention
        src2 = self.self_attn(src)
        src = src + src2
        src = self.norm1(src)

        # feed forward layer
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2
        src = self.norm2(src)
        
        return src


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        B_, L, C = x.shape
        qkv = rearrange(self.qkv(x), "B L (num nhead C) -> num B nhead L C",
                    num=3, nhead=self.num_heads, C=C // self.num_heads)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """
    Return an activation function given a string
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
