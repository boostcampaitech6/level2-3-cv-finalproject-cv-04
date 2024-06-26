"""
Transformer Encoder and Decoder with Progressive Rectangle Window Attention
"""
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .utils import *

# kwargs : enc_win_list = [(32, 16), (32, 16), (16, 8), (16, 8)]
# d_model=256, dropout=0.0, nhead=8, dim_feedforward=512,num_encoder_layers=4
class WinEncoderTransformer(nn.Module):
    """
    Transformer Encoder, featured with progressive rectangle window attention
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                 dim_feedforward=512, dropout=0.0,
                 activation="relu", 
                 **kwargs):
        super().__init__()
        # d_model=256, dropout=0.0, nhead=8, dim_feedforward=512,num_encoder_layers=4, activation='gelu'
        # encoder_layer = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward,
        #                                             dropout, pool_size=7, activation=activation),
        encoder_layer = nn.ModuleList([EncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation=activation),
                                        EncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation=activation)])
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, **kwargs)
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

        self.enc_win_list = kwargs['enc_win_list']  #[(32, 16), (16, 8)]
        self.return_intermediate = kwargs['return_intermediate'] if 'return_intermediate' in kwargs else False           

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, src, pos_embed, mask):
        # src.shape [8, 256, 32, 32]
        return self.encoder.forward(src)
    

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
        # self.layers = _get_clones(encoder_layer, num_layers)  # 인코더 레이어(블록)가 4개 들어감
        self.layers = encoder_layer
        self.num_layers = num_layers

        if 'return_intermediate' in kwargs:
            self.return_intermediate = kwargs['return_intermediate']
        else:
            self.return_intermediate = False
    # memory_win, mask=None, src_key_padding_mask=mask_win, pos=pos_embed_win, layer_idx=idx
    def single_forward(self, src, win_shape,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                layer_idx=0):
        
        output = src  # 윈도우화 한 이미지 특징
        # breakpoint()
        layer = self.layers[layer_idx]  # idx번째 인코더 레이어
        
        output = layer(output, win_shape, src_mask=mask,
                        src_key_padding_mask=src_key_padding_mask, pos=pos)        
        return output

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        intermediate = []
        output = src
        # [8, 256, 32, 32]
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
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.0, pool_size=3,
                 activation="relu"):
        super().__init__()
        
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.scaled_dot_attn = nn.MultiheadAttention
        
        self.poolformer = Pooling(pool_size=pool_size)
        self.linear1 = nn.Linear(d_model, dim_feedforward)  # 256, 512
        self.linear2 = nn.Linear(dim_feedforward, d_model)  # 512, 256
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    # memory_win, src_key_padding_mask=mask_win, pos=pos_embed_win, layer_idx=idx
    # memory_win, src_mask=None, src_key_padding_mask=mask_win, pos=pos_embed_win, layer_idx=idx
    # src: 윈도우화 되서 들어옴[512,16,256]
    def forward(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # [8, 256, 32, 32]
        src2 = self.poolformer(src)
        
        # residual connection & layer normalization
        src = src + src2
        # src: [8, 32, 32, 256]
        src = src.permute(0,2,3,1)
        src = self.norm1(src)
        # src.shape: [512, 16, 256]
        # feed forward layer (MLP)
        # src2.shape: [512, 16, 256]
        src2 = self.linear2(self.activation(self.linear1(src)))
        # residual connection & layer normalization
        src = src + src2
        src = self.norm2(src)
        src = src.permute(0,3,1,2)
        
        return src

    # PET: window 미리 사전 분할->Encoder로 들어옴->self_attention통과->입력과 output size 같으니
    # residual connection 후 LN,


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


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool2d(
            pool_size, stride=1, padding=pool_size//2, count_include_pad=False)

    def forward(self, x):
        """
        [B, C, H, W] = x.shape
        Subtraction of the input itself is added
        since the block already has a
        residual connection.
        """
        return self.pool(x)


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
