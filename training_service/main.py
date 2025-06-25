import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from fastapi import FastAPI, BackgroundTasks
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)

        # Decoder
        self.up1 = DoubleConv(256 + 128, 128)
        self.up2 = DoubleConv(128 + 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.pool(x1)
        x2 = self.down1(x2)
        x3 = self.pool(x2)
        x3 = self.down2(x3)

        # Decoder
        x = self.upsample(x3)
        x = torch.cat([x, x2], dim=1)
        x = self.up1(x)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up2(x)
        x = self.outc(x)
        return torch.sigmoid(x)


class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, img_size=256):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size

        # Получаем все изображения
        self.image_files = [f for f in os.listdir(images_dir)
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not self.image_files:
            raise ValueError(f"No images found in {images_dir}")

        # Проверяем соответствие масок
        self.samples = []
        for img_file in self.image_files:
            base_name = os.path.splitext(img_file)[0]
            mask_file = f"{base_name}.png"
            mask_path = os.path.join(masks_dir, mask_file)

            if not os.path.exists(mask_path):
                mask_file = f"{base_name}.jpg"
                mask_path = os.path.join(masks_dir, mask_file)

            if os.path.exists(mask_path):
                self.samples.append((img_file, mask_file))
            else:
                logger.warning(f"Mask not found for {img_file}")

        if not self.samples:
            raise ValueError("No valid image-mask pairs found")

        logger.info(f"Found {len(self.samples)} valid image-mask pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_file, mask_file = self.samples[idx]

        img_path = os.path.join(self.images_dir, img_file)
        mask_path = os.path.join(self.masks_dir, mask_file)

        # Загрузка и ресайз изображений
        image = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        mask = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size))

        # Преобразование в тензоры
        image = torch.from_numpy(np.array(image)).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0) / 255.0

        return image, mask


def train_model(images_dir, masks_dir):
    try:
        # Принудительно используем CPU и ограничиваем ресурсы
        device = torch.device("cpu")
        torch.set_num_threads(2)

        # Создаем датасет с уменьшенными изображениями
        dataset = SegmentationDataset(images_dir, masks_dir, img_size=256)

        # Уменьшенный batch_size
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)

        # Упрощенная модель
        model = UNet().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()

        logger.info("Starting training...")

        # Уменьшенное количество эпох
        for epoch in range(5):
            epoch_loss = 0.0
            for images, masks in dataloader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

                # Очистка памяти
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            avg_loss = epoch_loss / len(dataloader)
            logger.info(f"Epoch {epoch + 1}, Avg Loss: {avg_loss:.4f}")

        # Сохраняем модель
        torch.save(model.state_dict(), "segmentation_model.pth")
        logger.info("Training completed and model saved")

        return "Training completed successfully"

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


@app.post("/train")
async def start_training(background_tasks: BackgroundTasks):
    try:
        # Проверка наличия данных перед запуском
        if not os.path.exists("images") or not os.path.exists("masks"):
            raise FileNotFoundError("Images or masks directory not found")

        if not os.listdir("images") or not os.listdir("masks"):
            raise ValueError("One of the directories is empty")

        background_tasks.add_task(train_model, "images", "masks")
        return {"message": "Training started in background", "status": "success"}

    except Exception as e:
        return {"message": str(e), "status": "error"}