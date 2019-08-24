import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import h5py
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def scale(data):
    max_val = np.max(abs(data))
    print("Scaling with max value =", max_val)
    return data / max_val


print("Preparing data...")
hf = h5py.File("log/dataset", 'r')
shape = list(hf.get("shape"))
n_data = shape[0]
n_nbs = shape[1]
print("n_data =", n_data)
print("n_nbs =", n_nbs)
vars_in = 3
vars_out = 2
X = np.zeros((n_data, n_nbs, vars_in))
X[:, :, 0] = scale(np.array(hf.get("norms")).reshape((n_data, n_nbs)))
X[:, :, 1:3] = np.array(hf.get("units"))
y = scale(np.array(hf.get("dkernels")))
print(X[0])
print(y[0])
print("Done!")

inputs = layers.Input(shape=(n_nbs, vars_in))
x = layers.Dense(100, activation='tanh')(inputs)
x = layers.Dense(50, activation='tanh')(x)
x = layers.Dense(10, activation='tanh')(x)
outputs = layers.Dense(vars_out, activation='linear')(x)

model = models.Model(inputs=inputs, outputs=outputs)
tf.contrib.keras.utils.plot_model(model, to_file='kernel_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='loss', patience=5)
opt = optimizers.Adam(lr=1e-3)
model.compile(optimizer=opt,
              loss='mean_absolute_percentage_error',
              metrics=['mean_absolute_error'])
try:
    history = model.fit(X, y, epochs=20, batch_size=10, callbacks=[early_stop])
    plt.plot(history['epoch'], history['mean_absolute_error'])
    plt.show()
    model.save("nn_cont.h5")
    print("Model saved successfully.")
except KeyboardInterrupt:
    model.save("nn_cont.h5")
    print("Training interrupted manually.")
    print("Model saved successfully.")
