# import os
# import fiftyone.zoo as foz
# import fiftyone.utils.coco as fouc

# def download_subset():
#     print("Загрузка 1000 изображений из COCO-2017...")
#     # fiftyone скачает только 1000 картинок из val2017 (они уже размечены)
#     dataset = foz.load_zoo_dataset(
#         "coco-2017",
#         split="validation",
#         max_samples=1000,
#         dataset_name="tiny-coco-train"
#     )

#     os.makedirs("data/raw/images", exist_ok=True)
#     os.makedirs("data/raw/annotations", exist_ok=True)
    
#     print("Экспорт в формат COCO...")
#     # Экспортируем данные
#     dataset.export(
#         export_dir="data/raw",
#         dataset_type=fouc.COCODetectionDataset,
#         label_field="ground_truth",
#         overwrite=True
#     )
    
#     print("✅ Готово! Данные сохранены в папку data/raw/")

# if __name__ == "__main__":
#     download_subset()


# scripts/download_tiny_coco.py
import os
import fiftyone as fo
import fiftyone.types as fot

def export_existing():
    print("Подключаемся к скачанному датасету...")
    
    # Загружаем датасет, который fiftyone уже создал
    dataset = fo.load_dataset("tiny-coco-train")
    print(f"Найдено изображений: {len(dataset)}")
    
    # Создаем папки
    os.makedirs("data/raw/images", exist_ok=True)
    os.makedirs("data/raw/annotations", exist_ok=True)
    
    print("Экспорт в формат COCO...")
    
    # Правильный API для новой версии FiftyOne
    dataset.export(
        export_dir="data/raw",
        dataset_type=fot.COCODetectionDataset,
        label_field="ground_truth",
        overwrite=True
    )
    
    print("✅ Готово! Данные экспортированы в data/raw/")
    print("📁 Картинки: data/raw/data/")
    print("📄 Аннотации: data/raw/labels.json")

if __name__ == "__main__":
    export_existing()