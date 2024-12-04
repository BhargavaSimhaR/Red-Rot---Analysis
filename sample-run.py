import tensorflow as tf
from tensorflow.keras import datasets, layers, models # type: ignore
import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

# Define the path to the dataset
dataset_path = 'C:\\Users\\anuds\\OneDrive\\Desktop\\Images'

# Load the dataset with data augmentation
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224, 224),  # Adjust based on your model's input size
    batch_size=32  # Adjust based on your memory capacity
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),  # Adjust based on your model's input size
    batch_size=32  # Adjust based on your memory capacity
)

# Define the CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')  # Adjust the number of output classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_dataset, validation_data=validation_dataset, epochs=10)
