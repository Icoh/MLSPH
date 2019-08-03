import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.keras import layers, models, callbacks
import state_image as sti
import os
import h5py


data_dir = "./cnn_files/cnn_data/"
files = os.listdir(data_dir)
hdf_name = "training"
if hdf_name not in files:
    print("Executing state_image.py first!")
    sti.run(plot=False, skip=100)
else:
    print('Found', hdf_name, 'file.')
print("Starting CNN...")

hf = h5py.File(data_dir + hdf_name, 'r')
features = np.array(hf.get('features'))
labels = np.array(hf.get('labels'))
N = np.array(hf.get('number_of_particles'))
n_files, im_size, _, channels = features.shape
hf.close()

print("Feature file shape {}".format(features.shape))
print("Labels file shape {}".format(labels.shape))
print("Image size: {}x{}".format(im_size, im_size))
print("Nb. of files:{}. Nb. of particles:{}. Nb. of channels:{}.".format(n_files, N, channels))

X = features.reshape((n_files, im_size, im_size, channels))
y = labels.reshape(n_files, N)

# Use 90% of data for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print("Reshaped features and labels.")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


# Define Convolutional Network Model
model = models.Sequential()
model.add(layers.Conv2D(16, (3, 3), activation='elu',  input_shape=(im_size, im_size, channels)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(16, (3, 3), activation='elu'))
model.add(layers.MaxPooling2D((2, 2)))

# model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(int(2*N), activation='elu'))
# model.add(layers.Dropout(0.25))
model.add(layers.Dense(N, activation='linear'))

model.summary()

early_stop = callbacks.EarlyStopping(monitor='mean_absolute_error', patience=3)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

history = model.fit(X_train, y_train, epochs=10, batch_size=32,
                    callbacks=[early_stop], validation_data=(X_test, y_test))

model.evaluate(X_test, y_test)
model.save("model.h5")

