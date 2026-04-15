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
  

import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn, optim
from torch.utils.data import DataLoader

# transformations (IMPORTANT 🔥)
transform = transforms.Compose([
    transforms.Resize((150,150)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder("dataset/train", transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# modèle CNN simple
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(64*37*37, 128),
    nn.ReLU(),
    nn.Linear(128, 6)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# entraînement
epochs = 5

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss}")

# sauvegarde
torch.save(model, "daouda_model.pth")