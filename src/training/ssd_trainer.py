import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.dataset.dataset import UniversalCOCODataset, get_transforms, collate_fn
from src.models.ssd.model import SSDWrapper

class SSDTrainer:
    def __init__(self, img_dir, ann_file, epochs=20, batch_size=4, lr=0.001, momentum=0.9, weight_decay=0.0005):
        self.img_dir = img_dir
        self.ann_file = ann_file
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        self.device = torch.device('cpu')
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
        
        max_category_id = max(ann['category_id'] for ann in coco_data['annotations'])
        num_classes = max_category_id + 1
        
        print(f'Найдено классов в датасете: {num_classes} (максимальный category_id: {max_category_id})')
        
        self.dataset = UniversalCOCODataset(
            img_dir=self.img_dir, 
            ann_file=self.ann_file, 
            transforms=get_transforms(train=True)
        )
        
        self.dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0, 
            collate_fn=collate_fn
        )
        
        self.model = SSDWrapper(num_classes=num_classes, pretrained=True).to(self.device)

    def train(self):
        print('=' * 60)
        print('Обучение SSD')
        print('=' * 60)
        
        self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        
        os.makedirs('results/ssd/weights', exist_ok=True)
        os.makedirs('results/ssd/logs', exist_ok=True)
        
        log_file = open('results/ssd/logs/train_log.txt', 'w')
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0
            
            print(f'\nЭпоха {epoch+1}/{self.epochs}:')
            
            for i, (images, targets) in enumerate(tqdm(self.dataloader, desc=f'Epoch {epoch+1}')):
                images = list(image.to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                
                epoch_loss += losses.item()
                
            lr_scheduler.step()
            avg_loss = epoch_loss / len(self.dataloader)
            
            log_msg = f'Epoch {epoch+1}/{self.epochs} | Loss: {avg_loss:.4f} | LR: {lr_scheduler.get_last_lr()[0]:.6f}'
            print(log_msg)
            log_file.write(log_msg + '\n')
            log_file.flush()
            
            if (epoch + 1) % 5 == 0 or epoch == self.epochs - 1:
                torch.save(self.model.state_dict(), f'results/ssd/weights/epoch_{epoch+1}.pth')
                print(f'   Сохранены веса за эпоху {epoch+1}')
                
        log_file.close()
        torch.save(self.model.state_dict(), 'results/ssd/weights/best.pth')
        print('\n' + '=' * 60)
        print('Обучение SSD завершено!')
        print('=' * 60)