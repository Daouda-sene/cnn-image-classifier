import tensorflow as tf
from tensorflow.keras import layers, models

# Charger dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    image_size=(150, 150),
    batch_size=32
)

class_names = dataset.class_names
print("Classes:", class_names)

# -------- MODEL 1 --------
model1 = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(len(class_names), activation='softmax')
])

model1.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model1.fit(dataset, epochs=5)

model1.save("model1.h5")

# -------- MODEL 2 --------
model2 = models.Sequential([
    layers.Rescaling(1./255),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model2.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model2.fit(dataset, epochs=10)

model2.save("model2.h5")

print("✅ Models saved successfully!")