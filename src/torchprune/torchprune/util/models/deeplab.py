"""Module with Deeplab v3 segmentation networks."""

import torch
import mobilenetv2
import torch.nn as nn
import torchvision.models.segmentation as seg_models
from torchvision.models._utils import IntermediateLayerGetter
from torch.nn import functional as F


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

class _DeepLabV3(nn.Module):
    def __init__(self, backbone, classifier):
        super(_DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        
    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)
        x = self.classifier(features)
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=False)
        return x

class _DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[12, 24, 36]):
        super(_DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            _ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier( feature['out'] )

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class _ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(_ASPPConv, self).__init__(*modules)

class _ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(_ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class _ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(_ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(_ASPPConv(in_channels, out_channels, rate1))
        modules.append(_ASPPConv(in_channels, out_channels, rate2))
        modules.append(_ASPPConv(in_channels, out_channels, rate3))
        modules.append(_ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

def fcn_resnet50(num_classes):
    """Return torchvision fcn_resnet50 and wrap it in SingleOutNet."""
    return SingleOutNet(
        seg_models.fcn_resnet50(num_classes=num_classes, aux_loss=True)
    )


def fcn_resnet101(num_classes):
    """Return torchvision fcn_resnet101 and wrap it in SingleOutNet."""
    return SingleOutNet(
        seg_models.fcn_resnet101(num_classes=num_classes, aux_loss=True)
    )


def deeplabv3_resnet50(num_classes):
    """Return torchvision deeplabv3_resnet50 and wrap it in SingleOutNet."""
    return SingleOutNet(
        seg_models.deeplabv3_resnet50(num_classes=num_classes, aux_loss=True)
    )


def deeplabv3_resnet101(num_classes):
    """Return torchvision deeplabv3_resnet101 and wrap it in SingleOutNet."""
    return SingleOutNet(
        seg_models.deeplabv3_resnet101(num_classes=num_classes, aux_loss=True)
    )

def deeplab_v3_mobilenet_v2(num_classes):
    aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=True, output_stride=16)
    
    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24
    
    return_layers = {'high_level_features': 'out'}
    classifier = _DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = _DeepLabV3(backbone, classifier)
    return model
