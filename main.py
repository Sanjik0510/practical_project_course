import argparse
import sys
from pathlib import Path

# Добавляем корень проекта в путь
sys.path.append(str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(description='Обучение моделей детекции объектов')
    parser.add_argument('--model', type=str, required=True, 
                        choices=['yolo', 'yolov5', 'frcnn', 'ssd', 'retinanet'],
                        help='Модель для обучения: yolo, yolov5, frcnn, ssd, retinanet')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Количество эпох обучения')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Размер батча')
    
    args = parser.parse_args()
    
    print(f'\n{"="*60}')
    print(f'Запуск обучения модели: {args.model.upper()}')
    print(f'Эпохи: {args.epochs}, Batch size: {args.batch_size}')
    print(f'{"="*60}\n')
    
    if args.model == 'yolo':
        from src.training.yolov8_trainer import YOLOTrainer
        
        trainer = YOLOTrainer(
            coco_json='data/raw/annotations/labels.json',
            img_dir='data/raw/images',
            output_dir='data/processed/yolo_format',
            yaml_config='configs/yolov8_coco.yaml',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        trainer.run()
        
    elif args.model == 'yolov5':
        from src.training.yolov5_trainer import YOLOv5Trainer
        
        trainer = YOLOv5Trainer(
            coco_json='data/raw/annotations/labels.json',
            img_dir='data/raw/images',
            output_dir='data/processed/yolo_format',
            yaml_config='configs/yolov5_coco.yaml',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        trainer.run()
        
    elif args.model == 'frcnn':
        from src.training.frcnn_trainer import FasterRCNNTrainer
        
        trainer = FasterRCNNTrainer(
            img_dir='data/raw/images',
            ann_file='data/raw/annotations/labels.json',
            epochs=args.epochs,
            batch_size=2  # Для Faster R-CNN меньший batch size
        )
        trainer.train()
        
    elif args.model == 'ssd':
        from src.training.ssd_trainer import SSDTrainer
        
        trainer = SSDTrainer(
            img_dir='data/raw/images',
            ann_file='data/raw/annotations/labels.json',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        trainer.train()
        
    elif args.model == 'retinanet':
        from src.training.retinanet_trainer import RetinaNetTrainer
        
        trainer = RetinaNetTrainer(
            img_dir='data/raw/images',
            ann_file='data/raw/annotations/labels.json',
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        trainer.train()
    
    print(f'\n{"="*60}')
    print(f'Обучение модели {args.model.upper()} завершено!')
    print(f'{"="*60}\n')

if __name__ == '__main__':
    main()