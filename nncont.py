import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import normalize
import state_image as sti
import h5py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def scale(data):
    max_val = max(abs(data))
    print("Scaling with max value =", max_val)
    return data/max_val


print("Preparing data...")
hf = h5py.File("log/dataset", 'r')
shape = hf.get("shape")
n_data = shape[0]*shape[1]
X = np.zeros((n_data, 5))
X[:, 0] = np.array(hf.get("norms")).ravel()
X[:, 1:3] = np.array(hf.get("units")).reshape((-1, 2))
X[:, 3] = np.array(hf.get("veldiff")).reshape((-1, 2))[:, 0]
X[:, 4] = np.array(hf.get("veldiff")).reshape((-1, 2))[:, 1]
y = scale(np.array(hf.get("continuity")).reshape((-1, 1)))
print("Done!")

inputs = layers.Input(shape=(5,))
x = layers.Dense(100, activation='tanh')(inputs)
x = layers.Dense(200, activation='tanh')(x)
x = layers.Dense(200, activation='tanh')(x)
x = layers.Dense(200, activation='tanh')(x)
x = layers.Dense(100, activation='tanh')(x)
outputs = layers.Dense(1, activation='tanh')(x)

model = models.Model(inputs=inputs, outputs=outputs)
tf.contrib.keras.utils.plot_model(model, to_file='kernel_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='loss', patience=5)
opt = optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
try:
    history = model.fit(X, y, epochs=20, batch_size=100, callbacks=[early_stop])
    model.save("nn_cont.h5")
    print("Model saved successfully.")
except KeyboardInterrupt:
    model.save("nn_cont.h5")
    print("Training interrupted manually.")
    print("Model saved successfully.")
