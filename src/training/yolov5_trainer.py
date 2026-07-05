import os
import json
import shutil
from pathlib import Path
from src.utils.coco_to_yolo import convert_coco_to_yolo
from src.models.yolov5.model import YOLOv5Wrapper

class YOLOv5Trainer:
    def __init__(self, coco_json, img_dir, output_dir, yaml_config, epochs=20, batch_size=4):
        self.coco_json = coco_json
        self.img_dir = img_dir
        self.output_dir = output_dir
        self.yaml_config = yaml_config
        self.epochs = epochs
        self.batch_size = batch_size

    def prepare_data(self):
        print('Шаг 1: Конвертация аннотаций...')
        if not os.path.exists(self.output_dir):
            num_classes = convert_coco_to_yolo(self.coco_json, self.output_dir, self.img_dir)
        else:
            print('Данные уже сконвертированы')
            with open(self.coco_json, 'r') as f:
                coco_data = json.load(f)
            num_classes = len(coco_data['categories'])
        
        print(f'   Количество классов: {num_classes}')
        
        print('\nШаг 2: Подготовка изображений...')
        images_dir = Path(self.output_dir) / 'images'
        if not images_dir.exists():
            images_dir.mkdir(parents=True)
            for img_file in Path(self.img_dir).glob('*.jpg'):
                shutil.copy(img_file, images_dir / img_file.name)
            print(f'   Скопировано изображений: {len(list(images_dir.glob("*.jpg")))}')
        else:
            print('   Изображения уже на месте')
            
        return num_classes

    def run(self):
        print('=' * 60)
        print('Обучение YOLOv5 на COCO датасете')
        print('=' * 60)
        
        num_classes = self.prepare_data()
        
        print('\nШаг 3: Обучение модели...')
        model = YOLOv5Wrapper(
            num_classes=num_classes,
            img_size=416,
            pretrained=True
        )
        
        results = model.train_model(
            data_yaml=self.yaml_config,
            epochs=self.epochs,
            batch_size=self.batch_size,
            device='cpu'
        )
        
        print('\n' + '=' * 60)
        print('Обучение завершено!')
        print(f'   Результаты сохранены в: results/yolov5/train/')
        print('=' * 60)
        
        return results