import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

class FasterRCNNWrapper(nn.Module):
    def __init__(self, num_classes=91, pretrained=True):
        super(FasterRCNNWrapper, self).__init__()
        if pretrained:
            weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            self.model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.5)
        else:
            self.model = fasterrcnn_resnet50_fpn_v2(weights=None, box_score_thresh=0.5)
        
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        if self.training:
            if targets is None:
                raise ValueError("Targets must be provided during training")
            return self.model(images, targets)
        else:
            return self.model(images)