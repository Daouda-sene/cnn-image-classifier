import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import gradio as gr

# classes
classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# 🔥 MODELE
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

# 🔥 CHARGEMENT
model = CNNModel()
model.load_state_dict(torch.load("daouda_model.pth", map_location="cpu"))
model.eval()

# preprocessing
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.ToTensor()
])

# 🔥 PREDICTION
def predict(image):
    if image is None:
        return "❌ No image", {}

    image = image.convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)[0]

    top_class = classes[torch.argmax(probs).item()]
    confidences = {classes[i]: float(probs[i]) for i in range(len(classes))}

    return f"✅ Prediction: {top_class}", confidences


# 🎨 UI AVEC FOND BLEU
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="purple"
    ),
   css="""
html, body {
    background: linear-gradient(to right, #1e3c72, #2a5298) !important;
}

.gradio-container {
    background: transparent !important;
    color: white;
}

.gr-button {
    background-color: #ff7a18 !important;
    color: white !important;
    border-radius: 10px !important;
}

footer {
    display: none !important;
}
"""
) as demo:

    gr.Markdown("# 🌍 Image Classifier")
    gr.Markdown("Upload une image pour prédire la catégorie")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="📷 Image")
            btn = gr.Button("🚀 Prédire")

        with gr.Column():
            output_text = gr.Textbox(label="Résultat")
            output_label = gr.Label(label="Probabilités")

    btn.click(fn=predict, inputs=image_input, outputs=[output_text, output_label])

# lancer
demo.launch()