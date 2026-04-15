from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# Charger modèles
models = {
    "model1": tf.keras.models.load_model("model1.keras"),
    "model2": tf.keras.models.load_model("model2.keras")
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        file = request.files["image"]
        model_name = request.form["model"]
        model = models[model_name]

        print("Model choisi:", model_name)

        image_path = "static/test.jpg"
        file.save(image_path)

        # 🔥 CORRECTION ICI
        img = Image.open(image_path).convert("RGB").resize((150,150))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        class_index = np.argmax(pred)

        prediction = classes[class_index]
        confidence = round(np.max(pred) * 100, 2)

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image_path=image_path)

if __name__ == "__main__":
    app.run(debug=True)