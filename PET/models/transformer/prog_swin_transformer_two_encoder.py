"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .utils import *
from timm.models.layers import DropPath, to_2tuple

# kwargs : enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]
# d_model=256, dropout=0.0, nhead=8, dim_feedforward=512,num_encoder_layers=4
class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu", downsample=None, norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead

        self.enc_win_list = kwargs['enc_win_list']  #[(32, 16), (32, 16), (16, 8), (16, 8)]
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False           

        # d_model=256, dropout=0.0, nhead=8, dim_feedforward=512,num_encoder_layers=4, activation='gelu'
        # transformer encoder 생성
        # encoder_layer = EncoderLayer(d_model, nhead, dim_feedforward,
        #                                             dropout, activation)
        # 4개로 복사뿡뿡
        # self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, **kwargs)
        self.encoder = nn.ModuleList([
            SwinTransformerBlock(dim=d_model,
                                 num_heads=nhead, # nhead*2**i
                                 window_size = self.enc_win_list[i],
                                 shift_size=0, # if (i % 2 == 0) else (self.enc_win_list[i][0]//2, self.enc_win_list[i][1]//2), 
                                 ########## 03.09 - 1st: shift x, encoder 2개 (윈도우-PAtchMerging-윈도우) Not Yet
                                 ### 2nd: shift O, encoder 2개 (윈도우-PatchMerging-쉬프트) Not Yet
                                 ### 3nd: shift O, encoder 4개(윈도우-쉬프트-PatchMerging-윈도우-쉬프트) Not Yet
                                 downsample=downsample if (i % 2 == 0) else None,
                                 dim_feedforward=dim_feedforward,
                                 qkv_bias=True, qk_scale=None,
                                 encode_feats='4x' if (i % 2 == 0) else '8x',
                                 drop=dropout, attn_drop=0)
            for i in range(num_encoder_layers)])
        self._reset_parameters()
        
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, pos_embed, mask):
        bs, c, h, w = src.shape  # [8, 256, 32, 32]
        
        memory_list = []
        memory = src
        for encoder in self.encoder:
            memory = encoder(memory, pos_embed, mask)
            if self.return_intermediate:
                memory_list.append(memory)
        memory_ = memory_list if self.return_intermediate else memory
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
        # hs_win.shape: [2, 128, 64, 256]
        hs_tmp = [window_reverse_output(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        # hs.shape: [2, 1024, 8, 256]
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
        # [128, 64, 256], [128, 64, 256], [64, 128]
        memory_win = window_partition(src, int(self.dec_win_h/div_ratio), int(self.dec_win_w/div_ratio))
        pos_embed_win = window_partition(pos_embed, int(self.dec_win_h/div_ratio), int(self.dec_win_w/div_ratio))
        mask_win = window_partition(mask.unsqueeze(1), int(self.dec_win_h/div_ratio), int(self.dec_win_w/div_ratio))
        
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
                layer_idx=0):
        
        output = src  # 윈도우화 한 이미지 특징
        layer = self.layers[layer_idx]  # idx번째 인코더 레이어
        output = layer(output, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask, pos=pos)        
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
        
        # multihead-Attention 만들기 참 쉽네...
        # q,k,v의 embedding dimension, head만 넣어줘도 뚝딱 만들어 주네 ㅋㅋ
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = WindowAttention(d_model, )
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
                     pos: Optional[Tensor] = None):
        # query & key
        q = k = self.with_pos_embed(src, pos)
        
        # encoder self-attention
        # src2 = self.self_attention(q, k, value=src, attn_mask=src_mask,
        #                            key_padding_mask=src_key_padding_mask)[0]
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                                   key_padding_mask=src_key_padding_mask)[0]
        
        # residual connection & layer normalization
        src = src + src2
        src = self.norm1(src)

        # feed forward layer (MLP)
        src2 = self.linear2(self.activation(self.linear1(src)))
        
        # residual connection & layer normalization
        src = src + src2
        src = self.norm2(src)

        return src  # [vactorize window, batch size x windows per image, chennels]


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0,
                 activation="relu"):
        super().__init__()
        
        # attention layer
        # self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # feed forward layer
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        # self.layer_normalization1 = nn.LayerNorm(d_model)
        # self.layer_normalization2 = nn.LayerNorm(d_model)
        # self.layer_normalization3 = nn.LayerNorm(d_model)
        # self.activation = _get_activation_fn(activation)
        
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


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop)
        self.softmax = nn.Softmax(dim=-1)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, x, mask=None, pos = None, src_key_padding_mask: Optional[Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        q = k = self.with_pos_embed(x, pos)
        x = self.self_attn(q, k, value=x, attn_mask = mask, key_padding_mask=src_key_padding_mask)[0]

        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 dim_feedforward=256., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, downsample=None, encode_feats='4x',
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.dim_feedforward = dim_feedforward
        self.encode_feats = encode_feats
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim_feedforward, act_layer=act_layer, drop=drop)

        self.fused_window_process = fused_window_process

        if downsample is not None:
            # train에서,
            # FPN에서 32x32가 아닌, 64x64를 받아오고, swin의 PatchMerging을 통해 사이즈를 32x32로 줄인다.
            # FPN에서 이미 higher information을 모두 더하며 lower 까지 전달해 주었기 때문에, 64x64를 쓰는 것도 좋아보인다!
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, pos_embed, mask):
        
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)
        enc_win_h, enc_win_w = self.window_size
        pos_embed = pos_embed[self.encode_feats]
        mask = mask[self.encode_feats]
        shortcut = x
        x = self.norm1(x)
        x = x.permute(0,3,1,2)
        # cyclic shift
        if self.shift_size != 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
                shifted_pos = torch.roll(pos_embed, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(2, 3))
                shifted_mask = torch.roll(mask, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
                # partition windows
                x_windows = window_partition(shifted_x, enc_win_h, enc_win_w)
                pos_embed_win = window_partition(shifted_pos, enc_win_h, enc_win_w)
                mask_win = window_partition(shifted_mask, enc_win_h, enc_win_w)
            else:
                x_windows = window_partition(shifted_x, enc_win_h, enc_win_w)
                pos_embed_win = window_partition(shifted_pos, enc_win_h, enc_win_w)
                mask_win = window_partition(shifted_mask, enc_win_h, enc_win_w)
        else:
            shifted_x = x
            shifted_mask = mask
            shifted_pos = pos_embed
            # partition windows
            x_windows = window_partition(shifted_x, enc_win_h, enc_win_w)
            pos_embed_win = window_partition(shifted_pos, enc_win_h, enc_win_w)
            mask_win = window_partition(shifted_mask, enc_win_h, enc_win_w)
        # x_windows: [512, 16, 256]
        # W-MSA/SW-MSA
        # attn_windows.shape: [512, 16, 256]
        attn_windows = self.attn(x_windows, pos=pos_embed_win, src_key_padding_mask=mask_win)  # nW*B, window_size*window_size, C
        # merge windows -> nn.MultiheadAttention써서 merge되서 나온다!
        # attn_windows = enc_win_partition_reverse(attn_windows, enc_win_h, enc_win_w, H, W)
        
        # reverse cyclic shift
        if self.shift_size != 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(attn_windows, enc_win_h, enc_win_w, H, W)  
                x = torch.roll(shifted_x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(2, 3))
            else:
                x = window_reverse(attn_windows, enc_win_h, enc_win_w, H, W)
        else:
            shifted_x = window_reverse(attn_windows, enc_win_h, enc_win_w, H, W) 
            x = shifted_x
        # breakpoint()
        x = x.permute(0,2,3,1)
        # residual connection
        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        x = x.permute(0,3,1,2)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops    


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        # self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False) --> 원래는 위가 맞으나, pos_emb과 차원이 맞지 않는 문제로 조금 더 압축하는 방향으로 진행했다.
        self.reduction = nn.Linear(4 * dim, dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        B, C, H, W = x.shape
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.permute(0,2,3,1).contiguous()

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H//2, W//2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x).permute(0,3,1,2).contiguous()
        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"
    

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
