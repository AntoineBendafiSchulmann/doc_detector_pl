import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import Compose, Grayscale, Resize, ToTensor
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import glob
import numpy as np
import cv2

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
        self.mask_dir = mask_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = os.path.join(self.mask_dir, os.path.basename(image_path))
        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        mask_np = np.array(mask)
        contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_mask = np.zeros_like(mask_np)
        for contour in contours:
            area = cv2.contourArea(contour)
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4 and area > 2000:
                cv2.drawContours(filtered_mask, [approx], -1, (255), thickness=cv2.FILLED)
        mask = Image.fromarray(filtered_mask)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

class UNet(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.learning_rate = learning_rate
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, masks = batch
        outputs = self(images)
        loss = self.loss_fn(outputs, masks)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

def get_dataloaders(image_dir, mask_dir, batch_size=16, num_workers=4):
    transforms = Compose([
        Grayscale(),
        Resize((256, 256)),
        ToTensor()
    ])
    dataset = CustomImageDataset(image_dir, mask_dir, transform=transforms)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader

def export_onnx(model, file_name):
    dummy_input = torch.randn(1, 1, 256, 256).to(next(model.parameters()).device)
    torch.onnx.export(
        model,
        dummy_input,
        file_name,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=11
    )
    print(f"Modèle exporté en ONNX : {file_name}")

def main():
    image_dir = "data/images"
    mask_dir = "data/masks"
    models_dir = "models"

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"Dossier {models_dir} créé.")

    if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Le dossier {image_dir} ou {mask_dir} n'existe pas.")

    train_loader, val_loader = get_dataloaders(image_dir, mask_dir, batch_size=16, num_workers=4)

    model = UNet()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device utilisé pour le modèle : {device}")

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        logger=pl.loggers.TensorBoardLogger("logs/"),
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)

    pth_path = os.path.join(models_dir, "unet_model.pth")
    torch.save(model.state_dict(), pth_path)
    print(f"Modèle sauvegardé sous {pth_path}")

    onnx_path = os.path.join(models_dir, "unet_model.onnx")
    export_onnx(model, onnx_path)

if __name__ == "__main__":
    main()
