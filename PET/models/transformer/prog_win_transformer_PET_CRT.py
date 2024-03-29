"""
Transformer Encoder and Encoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
# from .utils import *
from .utils import *

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce

class HiLo(nn.Module):
    """
    HiLo Attention

    Paper: Fast Vision Transformers with HiLo Attention
    Link: https://arxiv.org/abs/2205.13213
    """               #256
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.head_dim = int(dim/num_heads)   # 256/8 = 32
        self.dim = dim  #  256

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)   # 8*0.5= 4
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * self.head_dim    # 4*32 = 128

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads # 4
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * self.head_dim    # 4*32 = 128

        # local window size. The `s` in our paper.
        self.ws = window_size
        
        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or self.head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias) # 256 128
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias) # 256 256
            self.l_proj = nn.Linear(self.l_dim, self.l_dim) # 128

        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:
            pool_size = 3
            self.pool = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2, count_include_pad=False)
            self.h_qkv = nn.Linear(self.dim, self.h_dim, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):
        B, H, W, C = x.shape #64,8,16,256
        h_group, w_group = H // self.ws, W // self.ws   # self.ws등분
        
        x = self.h_qkv(x) # 64 8 16 128
        C = x.shape[-1]
        x = Rearrange('B (h_group ws1) (w_group ws2) C -> B (h_group w_group C) ws1 ws2 ', B = B, h_group = h_group, w_group = w_group, ws1=self.ws, ws2=self.ws)(x)

        # x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3) # B, H/ws(h갯수), ws, W/ws(w갯수), ws, C -> B, H/ws, W/ws, ws, ws, C
        
        x = self.pool(x) - x
        x = Rearrange('B (h_group w_group C) ws1 ws2 -> B (h_group ws1) (w_group ws2) C ', B = B, C = C, h_group = h_group, w_group = w_group, ws1=self.ws, ws2=self.ws)(x)
        #B, h_group * self.ws, w_group * self.ws, self.h_dim)
        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape # 64 8 16 256
        
        #    64 8 16 128 // B H W linear
        #    64 8 16 128  ->    64  128        4                   32(128//4)
        # q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)
        #   64 4 128 32 -> B | l_heads | H*W | l_dim//l_heads 
        q = self.l_q(x)
        q = Rearrange('B H W (l_heads head_dim) -> B l_heads (H W) head_dim', B = B, H=H, W=W, l_heads=self.l_heads, head_dim=self.head_dim)(q)
        
        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W  # 64, 8, 16, 256 -> 64 256 8 16
            #  64, 256, 4, 8 > 64 256 32(h*w) >   64, 32, 256
            # x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = Rearrange('B C half_H half_W -> B (half_H half_W) C')(self.sr(x_)) # half인 이유 avgpool을 하고서 수치상으로 반토막났기때문
            
            #   64 32 256 >           64  32  2        4                 128//4=32  ->           2 64 4 32 32 | 갯수?, B, l_heads, half_HW, head_dim 
            # 원본; kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
            kv = Rearrange('B half_HW (kv l_heads head_dim) -> kv B l_heads half_HW head_dim', kv =2, l_heads=self.l_heads, head_dim=self.head_dim)(self.l_kv(x_))
            # 참고: 왜 head_dim이라고 썼나염? token dimension in Lo-Fi  | self.l_dim = self.l_heads * head_dim    # 4*32 = 128
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1] # 64, 4, 32, 32
        
        attn = (q @ k.transpose(-2, -1)) * self.scale   # B l_heads (H W) head_dim @ B, l_heads, head_dim, half_HW | 64 4 128 32 @ 64, 4, 32, 32
        attn = attn.softmax(dim=-1) # 64, 4, 128, 32 B l_heads (H W) half_HW
        
        # B l_heads (H W) head_dim -> B (H W) l_heads head_dim
        # 원본: x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim) # 64, 8, 16, 128
        x = Rearrange('B l_heads (H W) head_dim -> B H W (l_heads head_dim)', l_heads=self.l_heads, head_dim=self.head_dim, H=H, W=W)(attn @ v)
        x = self.l_proj(x)
        return x

    def forward(self, x, H):  # [B_, L, C] = [배치단위 총 윈도우 개수, 윈도우 벡터화(시퀀스), 채널 수]
        B, N, C = x.shape
        x = Rearrange('B (H W) C -> B H W C', B = B, H=H, W=N//H, C=C)(x)
        # x = x.reshape(B, H, N//H, C)
        

        if self.h_heads == 0:
            x = self.lofi(x)
            breakpoint()
            return x.reshape(N, B, C)

        if self.l_heads == 0:
            x = self.hifi(x)
            return x.reshape(N, B, C)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = Rearrange('B H W C -> B (H W) C', B = B, H=H, W=N//H, C=C)(x)
        # x = x.reshape(N, B, C)
        return x
    
    
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
            output = layer(src_win, src_pos_win, enc_win_h = enc_win_size[1],**kwargs)  # [L B' C]

            if idx+1 == self.num_layers and self.norm is not None:
                output = self.norm(output)
            
            # reverse encoder window
            output = window_reverse(output, enc_win_size[1], enc_win_size[0], H, W, "batch_first")  # [B C H W]
            
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu", **kwargs):
        super().__init__()
        self.self_attn = HiLo(d_model, nhead)
        
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
        
        self.enc_win_h = kwargs['enc_win_h']
        src2 = self.self_attn(src, self.enc_win_h)
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
