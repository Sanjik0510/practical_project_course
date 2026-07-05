import json
import os
from pathlib import Path

def convert_coco_to_yolo(coco_json_path, output_dir, img_dir):
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    images = {img['id']: img for img in coco_data['images']}
    categories = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
    
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    labels_dir = Path(output_dir) / 'labels'
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    converted_count = 0
    for img_id, anns in img_to_anns.items():
        img_info = images[img_id]
        img_width = img_info['width']
        img_height = img_info['height']
        
        img_filename = Path(img_info['file_name']).stem
        label_path = labels_dir / f'{img_filename}.txt'
        
        with open(label_path, 'w') as f:
            for ann in anns:
                category_id = ann['category_id']
                if category_id not in categories:
                    continue
                
                class_id = categories[category_id]
                x_min, y_min, bbox_width, bbox_height = ann['bbox']
                
                x_center = (x_min + bbox_width / 2) / img_width
                y_center = (y_min + bbox_height / 2) / img_height
                norm_width = bbox_width / img_width
                norm_height = bbox_height / img_height
                
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                norm_width = max(0, min(1, norm_width))
                norm_height = max(0, min(1, norm_height))
                
                f.write(f'{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n')
        
        converted_count += 1
    
    print(f'Конвертировано {converted_count} изображений в YOLO формат')
    print(f'Лейблы сохранены в: {labels_dir}')
    
    return len(categories)

if __name__ == '__main__':
    coco_json = 'data/raw/annotations/labels.json'
    output_dir = 'data/processed/yolo_format'
    img_dir = 'data/raw/images'
    
    num_classes = convert_coco_to_yolo(coco_json, output_dir, img_dir)
    print(f'Количество классов: {num_classes}')