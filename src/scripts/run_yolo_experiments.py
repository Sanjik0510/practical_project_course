import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.coco_to_yolo import convert_coco_to_yolo
from src.models.yolov8.model import YOLOv8Wrapper
import json
import os

def run_experiment(img_size, epochs=10, batch_size=4):
    print(f'\n{"="*60}')
    print(f'ЭКСПЕРИМЕНТ: YOLOv8 с img_size={img_size}')
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
        import shutil
        for img_file in Path(img_dir).glob('*.jpg'):
            shutil.copy(img_file, images_dir / img_file.name)
    
    # Обучение
    model = YOLOv8Wrapper(num_classes=num_classes, img_size=img_size, pretrained=True)
    
    results = model.train_model(
        data_yaml=yaml_config,
        epochs=epochs,
        batch_size=batch_size,
        device='cpu'
    )
    
    print(f'\n✅ Эксперимент с img_size={img_size} завершён!')
    return results

if __name__ == '__main__':
    # 3 эксперимента с разными размерами изображений
    img_sizes = [320, 416, 640]
    
    for img_size in img_sizes:
        run_experiment(img_size=img_size, epochs=10, batch_size=4)
    
    print(f'\n{"="*60}')
    print('ВСЕ ЭКСПЕРИМЕНТЫ YOLOv8 ЗАВЕРШЕНЫ!')
    print(f'{"="*60}')