# %%
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# %%
IMG_SIZE = 200
path_test = "Dataset"  # Folder structure: Dataset/Healthy and Dataset/Unhealthy
CATEGORIES = ["Healthy", "Unhealthy"]
training_data = []

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(path_test, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img))
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()

# %%
random.shuffle(training_data)

X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3).astype("float32") / 255.0
y = np.array(y)

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Reshape, LSTM, Dense

model = Sequential([
    Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu', padding='same'),
    MaxPooling2D((2,2)),
    Dropout(0.3),
    
    # Reshape for LSTM input: (batch_size, timesteps, features)
    Reshape((IMG_SIZE//4 * IMG_SIZE//4, 64)),  # Here 200/4=50 → 50*50 = 2500 timesteps
    LSTM(64),

    Dense(2, activation='softmax')  # Binary classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# %%
batch_size = 16
epochs = 5

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(X_test, y_test), verbose=1)

# %%
# Evaluation
score = model.evaluate(X_test, y_test, verbose=0)
print("Test Loss:", score[0])
print("Test Accuracy:", score[1])

# %%
# Prediction Function
def predict(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    input_data = np.expand_dims(resized_img, axis=0)
    preprocessed_input = input_data / 255.0

    prediction = model.predict(preprocessed_input)
    class_index = np.argmax(prediction[0])
    class_label = CATEGORIES[class_index]
    return class_label

# Example: print(predict("Dataset/Healthy/sample.jpg"))
