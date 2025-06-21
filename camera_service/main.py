import cv2
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import requests
import numpy as np
import threading
import logging
import time

app = FastAPI()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CameraProcessor:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.lock = threading.Lock()
        self.process = False
        self.inference_url = "http://localhost:8004/predict"

        # Запускаем поток для захвата кадров
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()

    def _capture_frames(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                time.sleep(0.1)
                continue

            with self.lock:
                self.frame = frame

            if self.process:
                self._process_frame(frame)

    def _process_frame(self, frame):
        try:
            # Преобразуем кадр в bytes
            _, img_encoded = cv2.imencode('.jpg', frame)
            files = {'file': ('frame.jpg', img_encoded.tobytes(), 'image/jpeg')}

            # Отправляем на inference сервис
            response = requests.post(self.inference_url, files=files)

            if response.status_code == 200:
                result = response.json()
                mask = cv2.imdecode(np.frombuffer(result['mask'], np.uint8), cv2.IMREAD_GRAYSCALE)

                # Здесь можно добавить обработку маски
                logger.info(f"Processed frame, mask size: {mask.shape}")
        except Exception as e:
            logger.error(f"Frame processing failed: {str(e)}")


camera = CameraProcessor()


@app.get("/stream")
async def video_stream(process: bool = False):
    camera.process = process

    def generate():
        while True:
            with camera.lock:
                if camera.frame is None:
                    continue

                frame = camera.frame.copy()

            # Кодируем кадр в JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
