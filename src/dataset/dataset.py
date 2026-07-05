import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.v2 as v2

class UniversalCOCODataset(Dataset):
    def __init__(self, img_dir, ann_file, transforms=None, max_samples=None):
        self.img_dir = img_dir
        self.transforms = transforms
        
        with open(ann_file, 'r') as f:
            coco_data = json.load(f)
            
        self.images = coco_data['images']
        self.annotations = coco_data['annotations']
        
        if max_samples and max_samples < len(self.images):
            self.images = self.images[:max_samples]
            
        self.img_to_anns = {}
        for ann in self.annotations:
            img_id = ann['image_id']
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f'️ Пропуск битого файла: {img_path}')
            return None, None
        
        boxes = []
        labels = []
        
        for ann in self.img_to_anns.get(img_id, []):
            xmin, ymin, w, h = ann['bbox']
            xmax = xmin + w
            ymax = ymin + h
            
            xmin = max(0, min(xmin, img.width))
            ymin = max(0, min(ymin, img.height))
            xmax = max(0, min(xmax, img.width))
            ymax = max(0, min(ymax, img.height))
            
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(ann['category_id'])

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.bool)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.bool)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": area,
            "iscrowd": iscrowd
        }

        if self.transforms:
            img, target = self.transforms(img, target)
            
        return img, target

def get_transforms(train=True):
    transforms = [
        v2.Resize((416, 416)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ]
    
    if train:
        transforms.insert(1, v2.RandomHorizontalFlip(p=0.5))
        
    return v2.Compose(transforms)

def collate_fn(batch):
    batch = [item for item in batch if item[0] is not None]
    if len(batch) == 0:
        return None, None
    return tuple(zip(*batch))