import torch
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
import cv2
import os

from training_service.main import UNet

app = FastAPI()

# Загрузка модели
model = UNet().cpu()  # Используем упрощенную архитектуру как при обучении
model.load_state_dict(torch.load("segmentation_model.pth", map_location='cpu'))
model.eval()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Чтение изображения
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Препроцессинг
    image = image.resize((256, 256))  # Такой же размер как при обучении
    image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    image_tensor = image_tensor.unsqueeze(0).cpu()

    # Предсказание
    with torch.no_grad():
        output = model(image_tensor)

    # Постобработка
    mask = (output.squeeze().numpy() > 0.5).astype(np.uint8) * 255
    mask = cv2.resize(mask, (image.width, image.height))  # Возвращаем к исходному размеру

    # Кодирование результата
    _, mask_encoded = cv2.imencode('.png', mask)
    return {"mask": mask_encoded.tobytes()}
