from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# classes du dataset Intel
classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# charger ton modèle
model = tf.keras.models.load_model("daouda_model.keras")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files["image"]
        file.save("test.jpg")

        # traitement image
        img = Image.open("test.jpg").resize((150,150))
        img = np.array(img)/255.0
        img = np.expand_dims(img, axis=0)

        # prédiction
        pred = model.predict(img)
        prediction = classes[np.argmax(pred)]

    return render_template("index.html", prediction=prediction)

# lancer le serveur
app.run(debug=True)