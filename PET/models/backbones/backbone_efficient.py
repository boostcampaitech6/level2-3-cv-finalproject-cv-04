from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from util.misc import NestedTensor
from .efficientnet import *
from ..position_encoding import build_position_encoding


class FeatsFusion(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, hidden_size=256, out_size=256, out_kernel=3):
        super(FeatsFusion, self).__init__()
        self.P5_1 = nn.Conv2d(C5_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P5_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P4_1 = nn.Conv2d(C4_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P4_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

        self.P3_1 = nn.Conv2d(C3_size, hidden_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(hidden_size, out_size, kernel_size=out_kernel, stride=1, padding=out_kernel//2)

    def forward(self, inputs):
        C3, C4, C5 = inputs
        C3_shape, C4_shape, C5_shape = C3.shape[-2:], C4.shape[-2:], C5.shape[-2:]
        C3_shape, C4_shape, C5_shape = [x * 2 for x in C3_shape], [x * 2 for x in C4_shape], [x * 2 for x in C5_shape]

        C5 = F.interpolate(C5, C5_shape)
        P5_x = self.P5_1(C5)
        P5_upsampled_x = F.interpolate(P5_x, C4_shape)
        P5_x = self.P5_2(P5_x)

        C4 = F.interpolate(C4, C4_shape)
        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = F.interpolate(P4_x, C3_shape)
        P4_x = self.P4_2(P4_x)

        C3 = F.interpolate(C3, C3_shape)
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        return [P3_x, P4_x, P5_x]


class BackboneBase_Efficient(nn.Module):
    def __init__(self, backbone: nn.Module, num_channels: int, name: str, return_interm_layers: bool):
        super().__init__()
        features = list(backbone.children())
        if return_interm_layers:
            layers = []
            if name == 'efficient_b0':
                layers.append(features[0])
                layers.append(features[1])
                layers.append(features[2][:2])
                self.body1 = nn.Sequential(*layers)
                self.body2 = features[2][2]
                self.body3 = features[2][3:5]
                self.body4 = features[2][5:]
        else:
            if name == 'vgg16_bn':
                self.body = nn.Sequential(*features[:44])  # 16x down-sample
        self.num_channels = num_channels
        self.return_interm_layers = return_interm_layers
        self.fpn = FeatsFusion(40, 112, 320, hidden_size=num_channels, out_size=num_channels, out_kernel=3)

    def forward(self, tensor_list: NestedTensor):
        feats = []
        if self.return_interm_layers:
            xs = tensor_list.tensors
            for idx, layer in enumerate([self.body1, self.body2, self.body3, self.body4]):
                xs = layer(xs)
                feats.append(xs)
                        
            # feature fusion
            features_fpn = self.fpn([feats[1], feats[2], feats[3]])
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


class Backbone_Efficient(BackboneBase_Efficient):
    """
    VGG backbone
    """
    def __init__(self, name: str, return_interm_layers: bool):
        if name == 'efficient_b0':
            backbone = efficient_b0(pretrained=True)
        elif name == 'efficient_b1':
            backbone = efficient_b1(pretrained=True)
        elif name == 'efficient_b2':
            backbone = efficient_b2(pretrained=True)
        elif name == 'efficient_b3':
            backbone = efficient_b3(pretrained=True)
        elif name == 'efficient_b4':
            backbone = efficient_b4(pretrained=True)
        elif name == 'efficient_b5':
            backbone = efficient_b5(pretrained=True)
        elif name == 'efficient_b6':
            backbone = efficient_b6(pretrained=True)
        elif name == 'efficient_b7':
            backbone = efficient_b7(pretrained=True)
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


def build_backbone_efficient(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone_Efficient(args.backbone, True)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model