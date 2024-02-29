import torch
import torch.nn as nn

import timm


__all__ = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
]

def resnet18(pretrained=False):
    return timm.create_model("resnet18", pretrained=pretrained, num_classes=0)

def resnet34(pretrained=False):
    return timm.create_model("resnet34", pretrained=pretrained, num_classes=0)

def resnet50(pretrained=False):
    return timm.create_model("resnet50", pretrained=pretrained, num_classes=0)

def resnet101(pretrained=False):
    return timm.create_model("resnet101", pretrained=pretrained, num_classes=0)

def resnet152(pretrained=False):
    return timm.create_model("resnet152", pretrained=pretrained, num_classes=0)