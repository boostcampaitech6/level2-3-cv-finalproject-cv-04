import torch
import torch.nn as nn

import timm


__all__ = [
    "efficient_b0", "efficient_b1", "efficient_b2", "efficient_b3",
    "efficient_b4", "efficient_b5", "efficient_b6", "efficient_b7"
]

def efficient_b0(pretrained=False):
    return timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)

def efficient_b1(pretrained=False):
    return timm.create_model("efficientnet_b1", pretrained=pretrained, num_classes=0)

def efficient_b2(pretrained=False):
    return timm.create_model("efficientnet_b2", pretrained=pretrained, num_classes=0)

def efficient_b3(pretrained=False):
    return timm.create_model("efficientnet_b3", pretrained=pretrained, num_classes=0)

def efficient_b4(pretrained=False):
    return timm.create_model("efficientnet_b4", pretrained=pretrained, num_classes=0)

def efficient_b5(pretrained=False):
    return timm.create_model("efficientnet_b5", pretrained=pretrained, num_classes=0)

def efficient_b6(pretrained=False):
    return timm.create_model("tf_efficientnet_b6", pretrained=pretrained, num_classes=0)

def efficient_b7(pretrained=False):
    return timm.create_model("tf_efficientnet_b7", pretrained=pretrained, num_classes=0)