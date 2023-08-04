import os
import numpy as np
import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from PIL import Image
import glob
import scipy
class_folders = glob.glob(os.path.join("train", "*"))


# Set the target image size
target_size = (224, 224)

# Load and preprocess the images using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
image_iterator = datagen.flow_from_directory(
    "train",
    target_size=target_size,
    batch_size=32,
    class_mode='sparse',
    shuffle=True
)

# Build the CNN architecture
model = tensorflow.keras.Sequential()
model.add(layers.Conv2D(64, (7, 7), activation='relu', input_shape=(target_size[0], target_size[1], 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(256, (5, 5), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(image_iterator.num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(image_iterator, epochs=60)

# Save the model
model.save("animal_detection_model.h5")
