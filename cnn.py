import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.keras import layers, models, callbacks
import state_image as sti
import os
import h5py


data_dir = "./cnn_files/cnn_data/"
files = os.listdir(data_dir)
hdf_name = "datafile"
if hdf_name not in files:
    print("Executing state_image.py first!")
    sti.run(plot=False, skip=100)
else:
    print('Found', hdf_name, 'file.')
print("Starting CNN...")

hf = h5py.File(data_dir + hdf_name, 'r')
n_data, n_nbs = np.array(hf.get('shape'))
print("NB count:", n_nbs)
print("Preparing features and labels...")
n_data = 1000
X = np.zeros((n_nbs, n_data, 2))
X[:, :, 0] = np.array(hf.get('posdiff'))[:n_data].transpose()
X[:, :, 1] = np.array(hf.get('veldiff'))[:n_data].transpose()
X = list(X)
y = list(np.array(hf.get('drho')).ravel())
hf.close()

# # Use 90% of data for training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# print("Reshaped features and labels.")
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)


# Define Neural Network Model
inputs, hidden = list(), list()
for n in range(n_nbs):
    inputs.append(layers.Input(shape=(2,)))
    hidden.append(layers.Dense(10, activation='relu')(inputs[n]))
x = layers.Concatenate()(hidden)
x = layers.Dense(100, activation='relu')(x)
out = layers.Dense(1, activation='linear')(x)

model = models.Model(inputs=inputs, outputs=out)
tf.contrib.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='mean_absolute_error', patience=5)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

history = model.fit(X, y, epochs=50, batch_size=32, callbacks=[early_stop])

index = -100
model.save("model.h5")
