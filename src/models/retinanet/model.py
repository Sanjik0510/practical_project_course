import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import retinanet_resnet50_fpn_v2, RetinaNet_ResNet50_FPN_V2_Weights

class RetinaNetWrapper(nn.Module):
    def __init__(self, num_classes=91, pretrained=True):
        super(RetinaNetWrapper, self).__init__()
        if pretrained:
            weights = RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = retinanet_resnet50_fpn_v2(weights=weights)
        else:
            self.model = retinanet_resnet50_fpn_v2(weights=None)
        
        if num_classes != 91:
            in_channels = self.model.backbone.out_channels
            num_anchors = self.model.head.classification_head.num_anchors
            self.model.head = torchvision.models.detection.retinanet.RetinaNetHead(
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