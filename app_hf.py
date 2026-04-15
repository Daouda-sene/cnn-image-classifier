import tensorflow as tf
import numpy as np
import gradio as gr
from PIL import Image

classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

model = tf.keras.models.load_model("model1.keras")

def predict(image):
    image = Image.fromarray(image).convert("RGB").resize((150,150))
    image = np.array(image)/255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)
    return classes[np.argmax(pred)]

gr.Interface(fn=predict, inputs=gr.Image(), outputs="label").launch()