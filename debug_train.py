import os
from training_service.main import train_model, SegmentationDataset  # Добавлен импорт SegmentationDataset

# Явно указываем пути
images_path = os.path.abspath("images")
masks_path = os.path.abspath("masks")

print("=== ДИАГНОСТИКА ДАННЫХ ===")
print(f"Images path: {images_path}")
print(f"Masks path: {masks_path}")

# Создаем датасет для проверки
try:
    dataset = SegmentationDataset(images_path, masks_path)
    print(f"\nFound {len(dataset)} valid image-mask pairs")

    # Проверка первых 5 пар
    print("\nПроверка первых 5 соответствий:")
    for i in range(min(5, len(dataset))):
        img_file, mask_file = dataset.samples[i]
        print(f"{i + 1}. Image: {img_file}")
        print(f"   Mask: {mask_file}")
        print(f"   Image exists: {os.path.exists(os.path.join(images_path, img_file))}")
        print(f"   Mask exists: {os.path.exists(os.path.join(masks_path, mask_file))}\n")

    # Запуск обучения
    print("\n=== ЗАПУСК ОБУЧЕНИЯ ===")
    train_model(images_path, masks_path)

except Exception as e:
    print(f"\nОШИБКА: {str(e)}")
    print("\nСоветы по устранению:")
    print("1. Проверьте соответствие имен файлов")
    print("2. Убедитесь, что для каждого .jpg есть .png")
    print("3. Проверьте права доступа к файлам")