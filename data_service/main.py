import os
from fastapi import FastAPI
from PIL import Image
import numpy as np
import cv2

app = FastAPI()


class DatasetLoader:
    def __init__(self, images_dir="images", masks_dir="masks"):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.mask_files = sorted(os.listdir(masks_dir))

        # Проверка соответствия изображений и масок
        assert len(self.image_files) == len(self.mask_files), "Количество изображений и масок не совпадает"
        for img, msk in zip(self.image_files, self.mask_files):
            assert os.path.splitext(img)[0] == os.path.splitext(msk)[0], f"Несоответствие имен: {img} и {msk}"

    def load_image_and_mask(self, idx):
        img_path = os.path.join(self.images_dir, self.image_files[idx])
        mask_path = os.path.join(self.masks_dir, self.mask_files[idx])

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))  # Градации серого

        return image, mask


dataset = DatasetLoader()


@app.get("/data/count")
async def get_data_count():
    return {"count": len(dataset.image_files)}


@app.get("/data/sample/{idx}")
async def get_sample(idx: int):
    if idx < 0 or idx >= len(dataset.image_files):
        return {"error": "Invalid index"}

    image, mask = dataset.load_image_and_mask(idx)

    # Конвертируем в base64 для демонстрации
    _, img_encoded = cv2.imencode('.jpg', image)
    _, mask_encoded = cv2.imencode('.png', mask)

    return {
        "image": img_encoded.tobytes(),
        "mask": mask_encoded.tobytes()
    }