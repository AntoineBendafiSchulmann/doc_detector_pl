import os
import torch
from torchvision.transforms import Compose, Resize, ToTensor
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(32),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(64),
            torch.nn.MaxPool2d(2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128)
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def test_model_and_crop(image_path, mask_path, model_path, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()  
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    image = Image.open(image_path).convert("L")
    mask = Image.open(mask_path).convert("L")

    transform = Compose([Resize((256, 256)), ToTensor()])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_mask = torch.sigmoid(output).squeeze().cpu().numpy() > 0.5

    predicted_mask_resized = Resize(image.size)(Image.fromarray(predicted_mask.astype(np.uint8) * 255))
    predicted_mask_np = np.array(predicted_mask_resized)
    coords = np.column_stack(np.where(predicted_mask_np > 0))
    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped_image = image.crop((x_min, y_min, x_max, y_max))
    else:
        cropped_image = image

    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(image, cmap="gray")
    axs[0].set_title("Image originale")
    axs[1].imshow(mask, cmap="gray")
    axs[1].set_title("Masque attendu")
    axs[2].imshow(predicted_mask, cmap="gray")
    axs[2].set_title("Masque prédit")
    axs[3].imshow(cropped_image, cmap="gray")
    axs[3].set_title("Image recadrée")

    base_name = os.path.basename(image_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{base_name}_cropped_comparison.png")
    plt.savefig(output_path)
    print(f"Résultat enregistré sous : {output_path}")
    plt.close()

test_model_and_crop(
    image_path="data/test_images/test_image2.png", 
    mask_path="data/masks/doc_7290.jpg",        
    model_path="models/unet_model.pth"
)
