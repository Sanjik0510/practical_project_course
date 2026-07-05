import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights

class SSDWrapper(nn.Module):
    def __init__(self, num_classes=91, pretrained=True):
        super(SSDWrapper, self).__init__()
        if pretrained:
            weights = SSD300_VGG16_Weights.DEFAULT
            self.model = ssd300_vgg16(weights=weights)
        else:
            self.model = ssd300_vgg16(weights=None)

        if num_classes != 91:
            in_channels = [512, 512, 512, 256, 256, 256]
            num_anchors = [4, 6, 6, 6, 4, 4]
            self.model.head = torchvision.models.detection.ssd.SSDHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes
            )

    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided during training")
            return self.model(images, targets)
        else:
            return self.model(images)