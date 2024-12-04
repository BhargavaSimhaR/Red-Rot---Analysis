# %%

# %%
import cv2
import numpy as np
import numpy as np
from tensorflow.keras.utils import to_categorical

from keras.models import Sequential
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import to_categorical
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import random
from numpy import *
from PIL import Image

# %%
IMG_SIZE = 200
path_test = "Dataset"
CATEGORIES = ["Healthy", "Unhealthy"]
training = []

def createTrainingData():
    for category in CATEGORIES:
        path = os.path.join(path_test, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training.append([new_array, class_num])

createTrainingData()

# %%
random.shuffle(training)

# %%
X =[]
y =[]
for features, label in training:
  X.append(features)
  y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

# %%
X = X.astype('float32')
X /= 255

Y = to_categorical(y, 4)
print(Y[100])
print(shape(Y))

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 4)
# Convert labels to NumPy array
y_train = np.array(y_train)
y_test = np.array(y_test)
nb_classes =2
# Convert labels to one-hot encoded format
y_train_encoded = to_categorical(y_train, num_classes=nb_classes)
y_test_encoded = to_categorical(y_test, num_classes=nb_classes)

# Convert input data to NumPy array
X_train = np.array(X_train)
X_test = np.array(X_test)

# Define and compile the model
model = tf.keras.Sequential([
    # Model layers...
])


# %%
# Assuming you have prepared your training and testing data and stored them in X_train, y_train, X_test, and y_test variables

batch_size = 16
nb_epochs = 5

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu,
                           input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(32, (3,3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(117, activation=tf.nn.relu),
    tf.keras.layers.Dense(2,  activation=tf.nn.softmax)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_data=(X_test, y_test))


# %%
score = model.evaluate(X_test, y_test, verbose = 0 )
print("Test Score: ", score[0])
print("Test accuracy: ", score[1])


# %%

def predict(image_path):
       # Replace with the path to your image

      img = cv2.imread(image_path)
      
      
      resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
      input_data = np.expand_dims(resized_img, axis=0)
  
      # Preprocess the input data
      preprocessed_input = input_data / 255.0  # Normalize the pixel values if needed


      # Make a prediction
      prediction = model.predict(preprocessed_input)
      class_index = np.argmax(prediction[0])  # Get the index of the class with the highest probability
      class_label = CATEGORIES[class_index]  # Get the corresponding class label from the CATEGORIES list
      return class_label



# %%

