"""Module with Deeplab v3 segmentation networks."""

import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models


class SingleOutNet(nn.Module):
    """A wrapper module to only return "out" from output dictionary in eval."""

    def __init__(self, network):
        """Initialize with the network that needs to be wrapped."""
        super().__init__()
        self.network = network

    def forward(self, x):
        """Only return the "out" of all the outputs."""
        if self.training or torch.is_grad_enabled():
            return self.network.forward(x)
        else:
            return self.network.forward(x)["out"]


def lraspp_mobilenet_v3_large(num_classes):
    """Return torchvision lraspp_mobilenet_v3_large and wrap it in SingleOutNet."""
    return SingleOutNet(
        seg_models.lraspp_mobilenet_v3_large(num_classes=num_classes, pretrained_backbone=True)
    )
