from fastapi import FastAPI, BackgroundTasks
from model_service.model import UNet
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import os
from PIL import Image

app = FastAPI()


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.filenames = [f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.filenames[idx])
        mask_path = os.path.join(self.masks_dir, self.filenames[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Преобразуем в тензоры и нормализуем
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0

        return image, mask


def train_model(images_dir, masks_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Создаем датасет и загрузчик
    dataset = SegmentationDataset(images_dir, masks_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Инициализируем модель
    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()

    # Процесс обучения
    for epoch in range(10):  # 10 эпох для примера
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Сохраняем модель
    torch.save(model.state_dict(), "segmentation_model.pth")
    return "Training completed"


@app.post("/train")
async def start_training(background_tasks: BackgroundTasks):
    background_tasks.add_task(train_model, "images", "masks")
    return {"message": "Training started in background"}