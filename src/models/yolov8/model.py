import torch
import torch.nn as nn
from ultralytics import YOLO

class YOLOv8Wrapper(nn.Module):
    def __init__(self, num_classes=80, img_size=416, pretrained=True):
        super(YOLOv8Wrapper, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        if pretrained:
            self.model = YOLO('yolov8n.pt')
        else:
            self.model = YOLO('yolov8n.yaml')
    
    def forward(self, images, targets=None):
        if self.training:
            pass
        results = self.model(images, verbose=False)
        return results
    
    def train_model(self, data_yaml, epochs=20, batch_size=4, device='cpu'):
        print(f'Запуск обучения YOLOv8...')
        print(f'   Эпохи: {epochs}')
        print(f'   Batch size: {batch_size}')
        print(f'   Image size: {self.img_size}')
        print(f'   Device: {device}')
        
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=self.img_size,
            batch=batch_size,
            device=device,
            workers=0,
            patience=5,
            plots=True,
            save=True,
            verbose=True,
            project=f'results/yolov8_exp_{self.img_size}',
            name='train',
            exist_ok=True
        )
        
        print('Обучение завершено!')
        return results
    
    def evaluate(self, data_yaml):
        metrics = self.model.val(
            data=data_yaml,
            imgsz=self.img_size,
            batch=4,
            device='cpu',
            workers=0
        )
        return metrics