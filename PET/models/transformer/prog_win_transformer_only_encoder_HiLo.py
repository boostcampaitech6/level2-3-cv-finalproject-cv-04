"""
Transformer Encoder and Encoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .utils import *

class HiLo(nn.Module):
    """
    HiLo Attention

    Paper: Fast Vision Transformers with HiLo Attention
    Link: https://arxiv.org/abs/2205.13213
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        #dim은 16이라고 가정
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)   # 2
        self.dim = dim  #  16

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)   # 8*0.5= 4
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim    # 4*2 = 8

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads # 4
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim    # 4*2 = 8

        # local window size. The `s` in our paper.
        self.ws = window_size

        if self.ws == 1:
            # ws == 1 is equal to a standard multi-head self-attention
            self.h_heads = 0
            self.h_dim = 0
            self.l_heads = num_heads
            self.l_dim = dim

        self.scale = qk_scale or head_dim ** -0.5

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)

    def hifi(self, x):
        B, H, W, C = x.shape #64,8,16,256
        h_group, w_group = H // self.ws, W // self.ws   # self.ws등분

        total_groups = h_group * w_group    # 이게 emb랑 비슷하지 않나?

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3) # B, H/ws, ws, W/ws, ws, C -> B, H/ws, W/ws, ws, ws, C
                    #64    4        2         8        2    256 -> 64 4 8 2 2 256
        # print(x.shape)
        # print(self.h_quk(x).shape) torch.Size([64, 4, 8, 2, 2, 384])
        # breakpoint()
        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim # 64,32,4,4,32
        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim) # 64 4 8 2 2 256
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim) # 64 8 16 128

        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape

        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)  # B, H, W, C -> B, C, H, W
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)
        x = self.l_proj(x)
        return x

    def forward(self, x, H):  # [L, B_, C] = [윈도우 벡터화(시퀀스 길이), 배치 단위 총 윈도우 개수, 엠베딩(채널)]
        N, B, C = x.shape
        # B, N, C = x.shape

        x = x.reshape(B, H, N//H, C)

        if self.h_heads == 0:
            x = self.lofi(x)
            return x.reshape(N, B, C)

        if self.l_heads == 0:
            x = self.hifi(x)
            return x.reshape(N, B, C)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(N, B, C)

        return x
    
    
########################################################################################



class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=2, 
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 return_intermediate_dec=False,
                 enc_win_w=16, enc_win_h=8,
                 **kwargs):
        super().__init__()
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)

        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm,
                                            return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.enc_win_w, self.enc_win_h = enc_win_w, enc_win_h
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_encoder_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encoder_forward(self, query_feats, query_embed, enc_win_h, enc_win_w, **kwargs): #bchw 8/256/sparse:32/ dense64
        """ 
        encoder forward during training
        """
        qH, qW = query_feats.shape[-2:] #[8, 256, 32, 32]
        # breakpoint()
        # window-rize query input
        # print(query_embed.shape) 1024,8,256
        query_embed_ = query_embed.permute(1,2,0).reshape(8, 256, qH, qW)
        query_embed_win = window_partition(query_embed_, window_size_h=enc_win_h, window_size_w=enc_win_w)
        tgt = window_partition(query_feats, window_size_h=enc_win_h, window_size_w=enc_win_w)

        # encoder attention
        self.enc_win_h = enc_win_h
        hs_win = self.encoder(tgt, query_pos=query_embed_win, enc_win_h = self.enc_win_h, **kwargs)
        hs_tmp = [window_partition_reverse(hs_w, enc_win_h, enc_win_w, qH, qW) for hs_w in hs_win]
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        return hs
    
    def encoder_forward_dynamic(self, query_feats, query_embed, enc_win_h, **kwargs):
        """ 
        encoder forward during inference
        """       
        # encoder attention
        tgt = query_feats
        self.enc_win_h = enc_win_h
        hs_win = self.encoder(tgt, query_pos=query_embed, enc_win_h = self.enc_win_h, **kwargs)
        num_layer, num_elm, num_win, dim = hs_win.shape
        hs = hs_win.reshape(num_layer, num_elm * num_win, dim)
        return hs
    
    def forward(self, pqs, **kwargs):
        query_embed, points_queries, query_feats, v_idx = pqs
        self.enc_win_w, self.enc_win_h = kwargs['dec_win_size']
        
        # dynamic encoder forward
        if 'test' in kwargs:
            hs = self.encoder_forward_dynamic(query_feats, query_embed, enc_win_h=self.enc_win_h, **kwargs)
            return hs
        else:
            # encoder forward
            hs = self.encoder_forward(query_feats, query_embed, self.enc_win_h, self.enc_win_w, **kwargs)
            return hs.transpose(1, 2)


class TransformerEncoder(nn.Module):
    """
    Base Transformer Encoder
    """
    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, query_pos: Optional[Tensor] = None, **kwargs):
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):
            output = layer(output, query_pos=query_pos, **kwargs)
            
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu"):
        super().__init__()
        
        self.self_attn = HiLo(d_model, nhead)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos: Optional[Tensor] = None, **kwargs):
        # query & key
        #q = k = self.with_pos_embed(tgt, query_pos)
        
        # encoder self attention
        self.enc_win_h = kwargs['enc_win_h']
        target2 = self.self_attn(tgt,self.enc_win_h)#(query=q, key=k, value=tgt)[0]
        
        # residual connection & layer normalization
        target = tgt + target2
        target = self.norm1(target)

        # feed forward layer
        target2 = self.linear2(self.activation(self.linear1(target)))
        
        # residual connection & layer normalization
        target = target + target2
        target = self.norm2(target)
        
        return target


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