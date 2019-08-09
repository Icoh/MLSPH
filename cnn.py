import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.keras import layers, models, callbacks, optimizers
import state_image as sti
import os
import h5py
import atexit

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
n_data, n_nbs, _ = np.array(hf.get('shape'))
n_data = 1000000
vars = 4
print("Samples:", n_data)
print("Neighbor count:", n_nbs)
print("Preparing features and labels...")
X = np.zeros((n_nbs, n_data, vars))
X[:, :, 0] = np.array(hf.get('posdiff'))[:n_data, :, 0].transpose()
X[:, :, 1] = np.array(hf.get('posdiff'))[:n_data, :, 1].transpose()
X[:, :, 2] = np.array(hf.get('veldiff'))[:n_data, :, 0].transpose()
X[:, :, 3] = np.array(hf.get('veldiff'))[:n_data, :, 1].transpose()
X = list(X)
y = list(np.array(hf.get('drho'))[:n_data].ravel())
hf.close()

# # Use 60% of data for training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
# print("Reshaped features and labels.")
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)


# Define Neural Network Model
inputs, hidden1, hidden2 = list(), list(), list()
for n in range(n_nbs):
    inputs.append(layers.Input(shape=(vars,)))
    hidden1.append(layers.Dense(10, activation='tanh', use_bias=False)(inputs[n]))
    hidden2.append(layers.Dense(10, activation='tanh')(hidden1[n]))
x = layers.Concatenate()(hidden2)
x = layers.Dense(250, activation='tanh')(x)
x = layers.Dense(50, activation='tanh')(x)
out = layers.Dense(1, activation='linear')(x)

model = models.Model(inputs=inputs, outputs=out)
tf.contrib.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='loss', patience=5)
opt = optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mean_absolute_error'])

history = model.fit(X, y, epochs=40, batch_size=100, callbacks=[early_stop])
atexit.register(model.save("model.h5"))
