import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.coco_to_yolo import convert_coco_to_yolo
from ultralytics import YOLO
import json
import os
import shutil
from pathlib import Path

def run_experiment(lr, epochs=10, batch_size=4):
    print(f'\n{"="*60}')
    print(f'ЭКСПЕРИМЕНТ: YOLOv8 с learning_rate={lr}')
    print(f'{"="*60}\n')
    
    coco_json = 'data/raw/annotations/labels.json'
    img_dir = 'data/raw/images'
    output_dir = 'data/processed/yolo_format'
    yaml_config = 'configs/yolov8_coco.yaml'
    
    # Конвертация данных (если ещё не сделано)
    if not os.path.exists(output_dir):
        num_classes = convert_coco_to_yolo(coco_json, output_dir, img_dir)
    else:
        with open(coco_json, 'r') as f:
            coco_data = json.load(f)
        num_classes = len(coco_data['categories'])
    
    # Копирование изображений
    images_dir = Path(output_dir) / 'images'
    if not images_dir.exists():
        images_dir.mkdir(parents=True)
        for img_file in Path(img_dir).glob('*.jpg'):
            shutil.copy(img_file, images_dir / img_file.name)
    
    # Обучение с разными learning rate
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data=yaml_config,
        epochs=epochs,
        imgsz=416,
        batch=batch_size,
        device='cpu',
        workers=0,
        patience=5,
        plots=True,
        save=True,
        verbose=True,
        project=f'results/yolov8_lr_{lr}',
        name='train',
        exist_ok=True,
        lr0=lr  # Начальный learning rate
    )
    
    print(f'\n✅ Эксперимент с lr={lr} завершён!')
    return results

if __name__ == '__main__':
    # 3 эксперимента с разными learning rate
    learning_rates = [0.01, 0.001, 0.0001]
    
    for lr in learning_rates:
        run_experiment(lr=lr, epochs=10, batch_size=4)
    
    print(f'\n{"="*60}')
    print('ВСЕ ЭКСПЕРИМЕНТЫ YOLOv8 (Learning Rate) ЗАВЕРШЕНЫ!')
    print(f'{"="*60}')