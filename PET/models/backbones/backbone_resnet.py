from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from util.misc import NestedTensor
from .resnet import *
from ..position_encoding import build_position_encoding


class FeatsFusion(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, hidden_size=256, out_size=256, out_kernel=3):
        super(FeatsFusion, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P4_1 = nn.Conv2d(C4_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P3_1 = nn.Conv2d(C3_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P2_1 = nn.Conv2d(C2_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P2_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

    def forward(self, inputs):
        C2, C3, C4, C5 = inputs
        C2_shape, C3_shape, C4_shape, C5_shape = C2.shape[-2:], C3.shape[-2:], C4.shape[-2:], C5.shape[-2:]
        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, C4_shape)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, C3_shape)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P4_upsampled_x + P3_x
        P3_upsampled_x = F.interpolate(P3_x, C2_shape)
        P3_x = self.P3_2(P3_x)
        
        P2_x = self.P2_1(C2)
        P2_x = P3_upsampled_x + P2_x
        P2_x = self.P2_2(P2_x)

        return [P2_x, P3_x, P4_x]


class BackboneBase_Resnet(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.children())
        
        self.body1 = nn.Sequential(*features[:5])
        self.body2 = features[5]
        self.body3 = features[6]
        self.body4 = features[7]
            
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        
        if name == 'resnet18': 
            self.fpn = FeatsFusion(64, 128, 256, 512, hidden_size=num_channels, out_size=num_channels, out_kernel=3)
        elif name == 'resnet34': 
            self.fpn = FeatsFusion(64, 128, 256, 512, hidden_size=num_channels, out_size=num_channels, out_kernel=3)
        elif name == 'resnet50': 
            self.fpn = FeatsFusion(256, 512, 1024, 2048, hidden_size=num_channels, out_size=num_channels, out_kernel=3)
        elif name == 'resnet101': 
            self.fpn = FeatsFusion(256, 512, 1024, 2048, hidden_size=num_channels, out_size=num_channels, out_kernel=3)
        elif name == 'resnet152': 
            self.fpn = FeatsFusion(256, 512, 1024, 2048, hidden_size=num_channels, out_size=num_channels, out_kernel=3)
        
    def forward(self, tensor_list: NestedTensor):
        feats = []
        if self.return_interm_layers:
            xs = tensor_list.tensors
            for idx, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                feats.append(xs)
                        
            # feature fusion
            features_fpn = self.fpn(feats)
            features_fpn_4x = features_fpn[0]
            features_fpn_8x = features_fpn[1]

            # get tensor mask
            m = tensor_list.mask
            assert m is not None
            mask_4x = F.interpolate(m[None].float(), size=features_fpn_4x.shape[-2:]).to(torch.bool)[0]
            mask_8x = F.interpolate(m[None].float(), size=features_fpn_8x.shape[-2:]).to(torch.bool)[0]

            out: Dict[str, NestedTensor] = {}
            out['4x'] = NestedTensor(features_fpn_4x, mask_4x)
            out['8x'] = NestedTensor(features_fpn_8x, mask_8x)
        else:
            xs = self.body(tensor_list)
            out.append(xs)

        return out


class Backbone_Resnet(BackboneBase_Resnet):
    """
    VGG backbone
    """
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'resnet18':
            backbone = resnet18(pretrained=True)
        elif name == 'resnet34':
            backbone = resnet34(pretrained=True)
        elif name == 'resnet50':
            backbone = resnet50(pretrained=True)
        elif name == 'resnet101':
            backbone = resnet101(pretrained=True)
        elif name == 'resnet152':
            backbone = resnet152(pretrained=True)
        num_channels = 256
        super().__init__(backbone, num_channels, name, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[NestedTensor] = {}
        pos = {}
        for name, x in xs.items():
            out[name] = x
            # position encoding
            pos[name] = self[1](x).to(x.tensors.dtype)
        return out, pos


def build_backbone_resnet(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_Resnet(args.backbone, True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model