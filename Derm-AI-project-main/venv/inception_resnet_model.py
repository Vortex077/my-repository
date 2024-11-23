import os
import json
import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow
from keras import layers
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.utils import custom_object_scope


# Define the CustomScaleLayer class (if not already defined elsewhere)
class CustomScaleLayer(layers.Layer):
    def __init__(self, scale_factor=1.0, **kwargs):
        super(CustomScaleLayer, self).__init__(**kwargs)
        self.scale_factor = scale_factor

    def call(self, inputs):
        return inputs * self.scale_factor

# Define the data augmentation parameters
seed = 42
train_datagen = ImageDataGenerator(
    zoom_range=0.5,
    rotation_range=0.4,
    shear_range=0.3,
    width_shift_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    preprocessing_function=preprocess_input
)
val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Load the training and validation data
train = train_datagen.flow_from_directory(directory=r"C:\Users\AVIGHYAT\dermno_copy\train", target_size=(256, 256), batch_size=32, seed=seed)
val = val_datagen.flow_from_directory(directory=r"C:\Users\AVIGHYAT\dermno_copy\val", target_size=(256, 256), batch_size=32, seed=seed)

# Get a sample of the training data
t_img, label = next(train)

# Plot the sample images
def plotImage(img_arr, label):
    for im, l in zip(img_arr, label):
        plt.figure(figsize=(5, 5))
        plt.imshow(im)
        plt.show()

plotImage(t_img[:3], label[:3])

# Load the pre-trained InceptionResNetV2 model
base_model = InceptionResNetV2(input_shape=(256, 256, 3), weights='imagenet', include_top=False)

for layer in base_model.layers[-30:]:  # Unfreeze the last 30 layers
    layer.trainable = True

# Add a new classification head to the model
X = GlobalAveragePooling2D()(base_model.output)
X = Dropout(0.5)(X)
X = Dense(1024, activation='relu')(X)
predictions = Dense(units=9, activation='softmax')(X)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy, metrics=['accuracy'])

# Define the callbacks
es = EarlyStopping(monitor='accuracy', min_delta=0.01, patience=5, verbose=1)
mc = ModelCheckpoint(filepath="./Dermno_RenseNet_02.keras", monitor='accuracy', verbose=1, save_best_only=True)
cb = [mc, es]

# Train the model
his = model.fit(train, steps_per_epoch=len(train), epochs=30, verbose=1, callbacks=cb, validation_data=val, validation_steps=16)

# Load the reference dictionary
ref = dict(zip(list(train.class_indices.values()), list(train.class_indices.keys())))
with open('class_indices.json', 'w') as f:
    json.dump(ref, f)

val_loss, val_acc = model.evaluate(val)
print(f"Validation Loss: {val_loss}, Validation Accuracy: {val_acc}")

