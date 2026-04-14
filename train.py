import tensorflow as tf

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset/",
    image_size=(150, 150),
    batch_size=32
)

print(dataset.class_names)