from fastapi import FastAPI, UploadFile, File, HTTPException
import torch
import numpy as np
from PIL import Image
import io
from model_service.model import UNet
import logging
from datetime import datetime
import cv2

app = FastAPI()

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка модели
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
try:
    model.load_state_dict(torch.load("segmentation_model.pth", map_location=device))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError("Failed to load model")


def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        return image.unsqueeze(0).to(device)
    except Exception as e:
        logger.error(f"Image preprocessing failed: {str(e)}")
        raise


def postprocess_mask(mask):
    mask = mask.squeeze().cpu().numpy()
    mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = datetime.now()

    try:
        # 1. Получение и обработка изображения
        contents = await file.read()
        logger.info(f"Received image for prediction, size: {len(contents)} bytes")

        # 2. Препроцессинг
        input_tensor = preprocess_image(contents)

        # 3. Предсказание
        with torch.no_grad():
            output = model(input_tensor)

        # 4. Постпроцессинг
        mask = postprocess_mask(output)

        # 5. Кодирование результата
        _, mask_encoded = cv2.imencode('.png', mask)
        mask_bytes = mask_encoded.tobytes()

        # Логирование времени выполнения
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Prediction completed in {processing_time:.2f} seconds")

        return {
            "status": "success",
            "processing_time": processing_time,
            "mask": mask_bytes
        }

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))