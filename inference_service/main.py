import torch
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, UploadFile, File
import cv2
import os

from fastapi.middleware.cors import CORSMiddleware
from training_service.main import UNet

app = FastAPI()

# Добавляем CORS middleware для фронтенд-запросов
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка модели
model = UNet().cpu()  # Используем упрощенную архитектуру как при обучении
model.load_state_dict(torch.load("/home/t1/PycharmProjects/cameraML/segmentation_model.pth", map_location='cpu'))
model.eval()


def preprocess_image(frame):
    """Преобразование кадра в тензор для модели"""
    image = Image.fromarray(frame).convert("RGB")
    image = image.resize((256, 256))
    image_tensor = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
    return image_tensor.unsqueeze(0).cpu()


def predict_mask(frame):
    """Предсказание маски для кадра"""
    with torch.no_grad():
        output = model(preprocess_image(frame))
    mask = (output.squeeze().numpy() > 0.5).astype(np.uint8) * 255
    return cv2.resize(mask, (frame.shape[1], frame.shape[0]))


def main():
    cap = cv2.VideoCapture(0)  # 0 - индекс камеры (обычно встроенная)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Получаем маску
        mask = predict_mask(frame)

        # Наложение маски на кадр (зелёный цвет)
        overlay = frame.copy()
        overlay[mask > 0] = [0, 255, 0]  # Закрашиваем маску зелёным

        # Вывод результата
        cv2.imshow("Camera", frame)
        cv2.imshow("Mask", mask)
        cv2.imshow("Segmentation", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Выход по 'q'
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
