import torch
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

model = torch.load("daouda_model.pth", map_location="cpu")
model.eval()

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

gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="label",
    title="Image Classifier",
    description="Upload une image"
).launch()