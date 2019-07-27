import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.keras import layers, models, callbacks
import state_image as sti
import matplotlib.pyplot as plt
import os
import h5py


data_dir = "./cnn_data/"
files = os.listdir(data_dir)
n_files = len(files)
hdf_name = "simulation"
if hdf_name not in files:
    print("Executing state_image.py first!")
    sti.run()
else:
    print("Found HDF5 {} file.".format(hdf_name))

hf = h5py.File(data_dir + hdf_name, 'r')
features = np.array(hf.get('velocity'))
labels = np.array(hf.get('density'))
N = np.array(hf.get('number_of_particles'))
n_files, im_size, _, channels = features.shape

print("Feature file shape {}".format(features.shape))
print("Labels file shape {}".format(labels.shape))
print("Image size: {}x{}".format(im_size, im_size))
print("Nb. of files:{}. Nb. of particles:{}. Nb. of channels:{}.".format(n_files, N, channels))

X = features.reshape(n_files, im_size, im_size, channels)
y = labels.reshape(n_files, N)

# features = ["x", "z", "xvel", "zvel"]
# labels = "density"
# features_df = df[features]
# labels_df = df[labels]
#
# X = features_df.values.reshape(n_files, N, 2, 2)
# y = labels_df.values.reshape(n_files, N)
# print("X shape: ", X.shape)

# We will use 80% of the data for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print("Reshaped features and labels.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# Define Convolutional Network Model
model = models.Sequential()
model.add(layers.Conv2D(16, (4, 4), activation='relu',  input_shape=(im_size, im_size, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (1, 1), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(384, (3, 3), activation='relu'))
# model.add(layers.MaxPooling2D((3, 3)))

model.add(layers.Flatten())
model.add(layers.Dense(2*N, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(N, activation='linear'))

model.summary()

early_stop = callbacks.EarlyStopping(monitor='mean_absolute_error', patience=5)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'accuracy'])

history = model.fit(X_train, y_train, epochs=20, callbacks=[early_stop])

model.evaluate(X_test, y_test)
