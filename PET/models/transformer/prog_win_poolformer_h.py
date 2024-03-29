"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers import to_2tuple
from .utils_pool import *

class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu", 
                 **kwargs):
        super().__init__()
        encoder_layer = EncoderLayer(d_model, dim_feedforward,
                                                    dropout, activation)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, **kwargs)
        self._reset_parameters()

        self.d_model = d_model

        self.enc_win_list = kwargs['enc_win_list']
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False           

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src):
        bs, c, h, w = src.shape  # [8, 256, 32, 32]
        
        memeory_list = []
        memeory = src
        for idx, enc_win_size in enumerate(self.enc_win_list):  # [(32, 16), (32, 16), (16, 8), (16, 8)]
            # encoder window partition
            enc_win_w, enc_win_h = enc_win_size
            memeory_win = enc_win_partition_p(memeory, enc_win_h, enc_win_w)  # [512, 16, 256]

            # encoder forward
            output = self.encoder.single_forward(memeory_win, layer_idx=idx)

            # reverse encoder window
            memeory = enc_win_partition_reverse_p(output, enc_win_h, enc_win_w, h, w)  # memory.shape = [8, 256, 32, 32]
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

        decoder_norm = GroupNorm(d_model)
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
    
    def decoder_forward(self, query_feats, memory_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        """ 
        decoder forward during training
        """
        bs, c, h, w = src_shape
        qH, qW = query_feats.shape[-2:]

        # window-rize query input
        tgt = window_partition_p(query_feats, window_size_h=dec_win_h, window_size_w=dec_win_w)

        # decoder attention
        hs_win = self.decoder(tgt, memory_win, **kwargs)
        hs_tmp = [window_partition_reverse_p(hs_w, dec_win_h, dec_win_w, qH, qW) for hs_w in hs_win]
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        return hs
    
    def decoder_forward_dynamic(self, query_feats, memory_win, dec_win_h, dec_win_w, src_shape, **kwargs):
        """ 
        decoder forward during inference
        """       
        # decoder attention
        tgt = query_feats
        hs_win = self.decoder(tgt, memory_win, **kwargs)
        num_layer, num_win, dim, win_h, win_w = hs_win.shape
        hs = hs_win.reshape(num_layer, num_win * win_h * win_w, dim)
        return hs
    
    def forward(self, src, pqs, **kwargs):
        bs, c, h, w = src.shape
        points_queries, query_feats, v_idx = pqs
        self.dec_win_w, self.dec_win_h = kwargs['dec_win_size']
        
        # window-rize memory input
        div_ratio = 1 if kwargs['pq_stride'] == 8 else 2
        memory_win = enc_win_partition_p(src, int(self.dec_win_h/div_ratio), int(self.dec_win_w/div_ratio))
        
        # dynamic decoder forward
        if 'test' in kwargs:
            memory_win = memory_win[v_idx, :, :, :]
            hs = self.decoder_forward_dynamic(query_feats, memory_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
            return hs
        else:
            # decoder forward
            hs = self.decoder_forward(query_feats, memory_win, self.dec_win_h, self.dec_win_w, src.shape, **kwargs)
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
    
    def single_forward(self, src,layer_idx=0):
        
        output = src  # 윈도우화 한 이미지 특징
        layer = self.layers[layer_idx]  # idx번째 인코더 레이어
        output = layer(output)        
        return output

    def forward(self, src):
        
        intermediate = []
        output = src
        for layer in self.layers:
            output = layer(output)
            
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

    def forward(self, tgt, memory,**kwargs):
        output = tgt
        intermediate = []
        for idx, layer in enumerate(self.layers):
            output = layer(output, memory)
            
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


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            3, stride=1, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.pool(x) - x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """
    def __init__(self, in_features, hidden_features=None, 
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(-1).unsqueeze(-1) * x \
            + self.bias.unsqueeze(-1).unsqueeze(-1)
        return x
    
    
class GroupNorm(nn.GroupNorm):
    """
    Group Normalization with 1 group.
    Input: tensor in shape [B, C, H, W]
    """
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
    

class AttentionGuidedPooling(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AttentionGuidedPooling, self).__init__()
        # 인코더의 특징 맵에 적용될 어텐션 가중치 계산을 위한 레이어
        self.query_conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(input_dim, output_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(input_dim, input_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tgt, src):
        if tgt.shape != src.shape:
            src = F.interpolate(src, tgt.shape[-2:])

        B, C, H, W = src.size()

        # 인코더 특징 맵에서 query, key, value 생성
        query = self.query_conv(tgt).view(B, -1, C)  # [B, H*W, C]
        key = self.key_conv(src).view(B, -1, C)  # [B, H*W, C]
        value = self.value_conv(src).view(B, -1, C)  # [B, H*W, C]

        # query와 key의 유사도 계산 (내적을 공간 차원 H*W에서 수행)
        attn_outputs = torch.bmm(query, key.transpose(1, 2))  # [B, H*W, H*W]
        attn_outputs = self.softmax(attn_outputs)

        # 어텐션 가중치를 사용하여 value 집약
        out = torch.bmm(attn_outputs, value)  # [B, H*W, C]
        out = out.view(B, C, H, W)  # [B, C, H, W]

        return out
    

class EncoderLayer(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, d_model, pool_size=3, mlp_ratio=2., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.norm1 = GroupNorm(d_model)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = GroupNorm(d_model)
        mlp_hidden_dim = int(d_model * 2)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, 
                       act_layer=nn.GELU, drop=drop)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((d_model)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((d_model)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    


class DecoderLayer(nn.Module):
    """
    Implementation of one PoolFormer block.
    --dim: embedding dim
    --pool_size: pooling size
    --mlp_ratio: mlp expansion ratio
    --act_layer: activation
    --norm_layer: normalization
    --drop: dropout rate
    --drop path: Stochastic Depth, 
        refer to https://arxiv.org/abs/1603.09382
    --use_layer_scale, --layer_scale_init_value: LayerScale, 
        refer to https://arxiv.org/abs/2103.17239
    """
    def __init__(self, d_model, pool_size=3, mlp_ratio=2., 
                 act_layer=nn.GELU, norm_layer=GroupNorm, 
                 drop=0., drop_path=0., 
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.norm1 = GroupNorm(d_model)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = GroupNorm(d_model)
        mlp_hidden_dim = int(d_model * 2)
        self.mlp = Mlp(in_features=d_model, hidden_features=mlp_hidden_dim, 
                       act_layer=nn.GELU, drop=drop)
        self.attention_guided_pooling = AttentionGuidedPooling(d_model, d_model)

        # The following two techniques are useful to train deep PoolFormers.
        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((d_model)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((d_model)), requires_grad=True)
            self.layer_scale_3 = nn.Parameter(
                layer_scale_init_value * torch.ones((d_model)), requires_grad=True)
            
        
    def forward(self, tgt, memory):

        if self.use_layer_scale:
            tgt2 = tgt + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(tgt)))
            # 인코더의 어텐션 맵을 사용하여 디코더 입력에 대한 가이드 풀링 수행
            guided_features = self.attention_guided_pooling(tgt, memory)
            
            # 가이드된 특징 맵과 디코더의 입력을 결합
            # if tgt2.shape != guided_features.shape:
            #     tgt3 = tgt2 + F.interpolate(guided_features, tgt2.shape[-2:])
            # else:
            tgt3 = tgt2 + guided_features
            
            # 결합된 특징 맵에 풀링 적용
            tgt3 = tgt2 + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(self.norm1(tgt3)))
            
            tgt3 = tgt3 + self.drop_path(
                self.layer_scale_3.unsqueeze(-1).unsqueeze(-1)
                * self.mlp(self.norm2(tgt3)))
        return tgt3


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_encoder_p(args, **kwargs):
    return WinEncoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        # activation="gelu",  #### CHANGE
        **kwargs,
    )


def build_decoder_p(args, **kwargs):
    return WinDecoderTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
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