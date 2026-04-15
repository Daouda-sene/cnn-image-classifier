import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# 🔥 RECONSTRUIRE LE MODELE (IMPORTANT)
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 36 * 36, 128),
            nn.ReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# 🔥 Charger correctement
model = CNNModel()
model.load_state_dict(torch.load("daouda_model.pth", map_location="cpu"))
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor()
])

def predict(image):
    if image is None:
        return "No image"

    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)

    return classes[predicted.item()]

gr.Interface(fn=predict, inputs=gr.Image(type="pil"), outputs="label").launch()