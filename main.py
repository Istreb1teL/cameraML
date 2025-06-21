from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import requests
import io
from PIL import Image
import numpy as np
import cv2

app = FastAPI()

# Настройки CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Конфигурация сервисов
SERVICES = {
    "data": "http://localhost:8001",
    "training": "http://localhost:8002",
    "camera": "http://localhost:8003"
}


@app.get("/")
async def root():
    return {"message": "Table Object Segmentation API"}


@app.post("/predict")
async def predict_segmentation(file: UploadFile = File(...)):
    # 1. Получаем изображение
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    # 2. Отправляем в сервис обработки (здесь будет логика предсказания)
    # В реальности здесь будет запрос к inference сервису

    # 3. Возвращаем результат (заглушка)
    return {
        "original_image": contents,
        "segmentation_mask": contents  # В реальности здесь будет маска
    }


@app.get("/start_training")
async def start_training():
    response = requests.post(f"{SERVICES['training']}/train")
    return response.json()