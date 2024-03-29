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
    
    def encoder_forward(self, query_feats, query_embed, enc_win_h, enc_win_w, **kwargs):
        """ 
        encoder forward during training
        """
        qH, qW = query_feats.shape[-2:]
        # breakpoint()
        # window-rize query input
        query_embed_ = query_embed.permute(1,2,0).reshape(8, 256, qH, qW)
        query_embed_win = window_partition(query_embed_, window_size_h=enc_win_h, window_size_w=enc_win_w)
        tgt = window_partition(query_feats, window_size_h=enc_win_h, window_size_w=enc_win_w)

        # encoder attention
        hs_win = self.encoder(tgt, query_pos=query_embed_win, **kwargs)
        hs_tmp = [window_reverse_output(hs_w, enc_win_h, enc_win_w, qH, qW) for hs_w in hs_win]
        hs = torch.vstack([hs_t.unsqueeze(0) for hs_t in hs_tmp])
        return hs
    
    def encoder_forward_dynamic(self, query_feats, query_embed, **kwargs):
        """ 
        encoder forward during inference
        """       
        # encoder attention
        tgt = query_feats
        hs_win = self.encoder(tgt, query_pos=query_embed, **kwargs)
        num_layer, num_elm, num_win, dim = hs_win.shape
        hs = hs_win.reshape(num_layer, num_elm * num_win, dim)
        return hs
    
    def forward(self, pqs, **kwargs):
        query_embed, points_queries, query_feats, v_idx = pqs
        self.enc_win_w, self.enc_win_h = kwargs['dec_win_size']
        
        # dynamic encoder forward
        if 'test' in kwargs:
            hs = self.encoder_forward_dynamic(query_feats, query_embed, **kwargs)
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
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)
        

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, query_pos: Optional[Tensor] = None, **kwargs):
        # query & key
        q = k = self.with_pos_embed(tgt, query_pos)
        
        # encoder self attention
        target2 = self.self_attn(query=q, key=k, value=tgt)[0]
        
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