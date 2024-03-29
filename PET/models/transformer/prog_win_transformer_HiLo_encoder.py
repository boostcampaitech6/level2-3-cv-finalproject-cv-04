"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .utils import *
import math

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
        B, H, W, C = x.shape
        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

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

    def forward(self, x, H):  # [B_, L, C] = [배치단위 총 윈도우 개수, 윈도우 벡터화(시퀀스), 채널 수]
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
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu", 
                 **kwargs):
        super().__init__()
        encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation)
        
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, **kwargs)
        self._reset_parameters()
        
        self.enc_win_list = kwargs['enc_win_list']
        self.d_model = d_model
        self.nhead = nhead

        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False           

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, pos_embed, mask):
        bs, c, h, w = src.shape  # [8, 256, 32, 32]
        
        memeory_list = []
        memeory = src
        for idx, enc_win_size in enumerate(self.enc_win_list):  # [(32, 16), (32, 16), (16, 8), (16, 8)]
            # encoder window partition
            enc_win_w, enc_win_h = enc_win_size
            memeory_win, pos_embed_win, mask_win  = enc_win_partition(memeory, pos_embed, mask, enc_win_h, enc_win_w)  # [512, 16, 256]

            # encoder forward
            output = self.encoder.single_forward(memeory_win, src_key_padding_mask=mask_win, pos=pos_embed_win, layer_idx=idx, enc_win_h = enc_win_h )

            # reverse encoder window
            memeory = enc_win_partition_reverse(output, enc_win_h, enc_win_w, h, w)  # memory.shape = [8, 256, 32, 32]
            if self.return_intermediate:
                memeory_list.append(memeory)
        memory_ = memeory_list if self.return_intermediate else memeory
        return memory_


class WinDecoderTransformer(nn.Module):
    """
    Transformer Decoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_decoder_layers=2, 
                 dim_feedforward=512, dropout=0.0,
                 activation="relu",
                 return_intermediate_dec=False,
                 dec_win_w=16, dec_win_h=8,
                 ):
        super().__init__()
        decoder_layer = DecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)

        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                            return_intermediate=return_intermediate_dec)
        self._reset_parameters()

        self.dec_win_w, self.dec_win_h = dec_win_w, dec_win_h
        self.d_model = d_model
        self.nhead = nhead
        self.num_layer = num_decoder_layers

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def decoder_forward(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        """ 
        decoder forward during training
        """
        bs, c, h, w = src_shape
        qH, qW = query_feats.shape[-2:]

        # window-rize query input
        query_embed_ = query_embed.permute(1,2,0).reshape(bs, c, qH, qW)
        query_embed_win = window_partition(query_embed_, window_size_h=dec_win_h, window_size_w=dec_win_w)
        tgt = window_partition(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)

        # decoder attention
        hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win, 
                                                                        query_pos=query_embed_win, **kwargs)
        hs_tmp = [window_partition_reverse(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        return hs
    
    def decoder_forward_dynamic(self, query_feats, query_embed, memory_win, pos_embed_win, mask_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        """ 
        decoder forward during inference
        """       
        # decoder attention
        tgt = query_feats
        hs_win = self.decoder(tgt, memory_win, memory_key_padding_mask=mask_win, pos=pos_embed_win, 
                                                                        query_pos=query_embed, **kwargs)
        num_layer, num_elm, num_win, dim = hs_win.shape
        hs = hs_win.reshape(num_layer, num_elm * num_win, dim)
        return hs
    
    def forward(self, src, pos_embed, mask, pqs, **kwargs):
        bs, c, h, w = src.shape
        query_embed, points_queries, query_feats, v_idx = pqs
        self.dec_win_w, self.dec_win_h = kwargs['dec_win_size']
        
        # window-rize memory input
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        memory_win, pos_embed_win, mask_win = enc_win_partition(src, pos_embed, mask, 
                                                    int(self.dec_win_h/div_ratio), int(self.dec_win_w/div_ratio))
        
        # dynamic decoder forward
        if 'test' in kwargs:
            memory_win = memory_win[:,v_idx]
            pos_embed_win = pos_embed_win[:,v_idx]
            mask_win = mask_win[v_idx]
            hs = self.decoder_forward_dynamic(query_feats, query_embed, 
                                    memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
            return hs
        else:
            # decoder forward
            hs = self.decoder_forward(query_feats, query_embed, 
                                    memory_win, pos_embed_win, mask_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
            return hs.transpose(1, 2)
        

class TransformerEncoder(nn.Module):
    """
    Base Transformer Encoder
    """
    def __init__(self, encoder_layer, num_layers, **kwargs):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)  # 인코더 레이어(블록)가 4개 들어감
        self.num_layers = num_layers

        if 'return_intermediate' in kwargs:
            self.return_intermediate = kwargs['return_intermediate']
        else:
            self.return_intermediate = False
    
    def single_forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                layer_idx=0,
                **kwargs):
        
        output = src  # 윈도우화 한 이미지 특징
        layer = self.layers[layer_idx]  # idx번째 인코더 레이어
        self.enc_win_h = kwargs['enc_win_h']
        output = layer(output, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask, pos=pos, enc_win_h=self.enc_win_h)        
        return output

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        intermediate = []
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            
            if self.return_intermediate:
                intermediate.append(output)
        
        if self.return_intermediate:
            return intermediate

        return output


class TransformerDecoder(nn.Module):
    """
    Base Transformer Decoder
    """
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                **kwargs):
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):
            output = layer(output, memory, tgt_mask=tgt_mask,
                        memory_mask=memory_mask,
                        tgt_key_padding_mask=tgt_key_padding_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                        pos=pos, query_pos=query_pos)
            
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
        # attention layers
        # self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.scaled_dot_attn = nn.MultiheadAttention

        # feed forward layer
        # self.linear1 = nn.Linear(d_model, dim_feedforward)  # 256, 512
        # self.linear2 = nn.Linear(dim_feedforward, d_model)  # 512, 256
        # self.layer_normalization1 = nn.LayerNorm(d_model)
        # self.layer_normalization2 = nn.LayerNorm(d_model)
        # self.activation = _get_activation_fn(activation)
        self.self_attn = HiLo(d_model, nhead)#nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.scaled_dot_attn = nn.MultiheadAttention
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 256, 512
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 512, 256
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None, **kwargs):
        # query & key
        #q = k = self.with_pos_embed(src, pos)
        #print(q.shape)
        #print(src.shape)
        # encoder self-attention
        # src2 = self.self_attention(q, k, value=src, attn_mask=src_mask,
        #                            key_padding_mask=src_key_padding_mask)[0]
        self.enc_win_h = kwargs['enc_win_h']
        src2 = self.self_attn(src,self.enc_win_h)
        # residual connection & layer normalization
        src = src + src2
        src = self.norm1(src)

        # feed forward layer
        src2 = self.linear2(self.activation(self.linear1(src)))
        
        # residual connection & layer normalization
        src = src + src2
        src = self.norm2(src)
        
        return src  # [vactorize window, batch size x windows per image, chennels]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu"):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     ):
        # query & key
        q = k = self.with_pos_embed(tgt, query_pos)
        # decoder self attention
        # target2 = self.self_attention(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        # target = tgt + target2
        # target = self.layer_normalization1(target)
        target2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        target = tgt + target2
        target = self.norm1(target)
        

        # decoder cross attention
        # target2 = self.cross_attention(query=self.with_pos_embed(target, query_pos),
        #                             key=self.with_pos_embed(memory, pos),
        #                             value=memory, attn_mask=memory_mask,
        #                             key_padding_mask=memory_key_padding_mask)[0]
        # target = target + target2
        # target = self.layer_normalization2(target)
        target2 = self.multihead_attn(query=self.with_pos_embed(target, query_pos),
                                    key=self.with_pos_embed(memory, pos),
                                    value=memory, attn_mask=memory_mask,
                                    key_padding_mask=memory_key_padding_mask)[0]
        target = target + target2
        target = self.norm2(target)

        # feed forward layer
        target2 = self.linear2(self.activation(self.linear1(target)))
        
        # residual connection & layer normalization
        target = target + target2
        target = self.norm3(target)
        
        return target


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_encoder(args, **kwargs):
    return WinEncoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        activation="gelu",  #### CHANGE
        **kwargs,
    )


def build_decoder(args, **kwargs):
    return WinDecoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
    )


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
