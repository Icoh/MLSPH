import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import scipy.spatial as sp
import scipy.linalg as lin
import matplotlib.pyplot as plt
import time
from tensorflow.contrib.keras import layers, models, callbacks, optimizers
import tensorflow as tf


def minmax(data):
    max_val = np.max(data)
    min_val = np.min(data)

    def scaler(array):
        print("Scaling with max={},  min={}".format(max_val, min_val))
        return (array - min_val) / (max_val - min_val)

    def rescaler(array):
        print("Rescaling with max={},  min={}".format(max_val, min_val))
        return array * (max_val-min_val) + min_val
    return (data-min_val)/(max_val-min_val), scaler, rescaler


def scale(data):
    min_val = np.min(data)

    def scaler(array):
        print("Scaling with max={}".format(min_val))
        return array-min_val

    def rescaler(array):
        print("Rescaling with max={}".format(min_val))
        return array+min_val
    return data-min_val, scaler, rescaler


def gaussian(r, unit_vect, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g #*(-unit_vect)
    return g, dg


h = 0.009231
support = 3
samples = int(2e6)
print("Generating {} samples... ".format(samples))
norms = np.random.rand(samples, 1)*support*h
units = normalize(np.random.rand(samples, 2) * np.random.choice([-1, 0, 1], (samples, 2)))
posdiff = norms * units
kn, dkn = gaussian(norms, units, h)
print("Done!")

X, sc_X, rsc_X = minmax(norms)
y, sc_y, rsc_y = minmax(dkn)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train[:5])
print(X_train.shape, np.min(X), np.max(X))
print(y_train[:5])
print(y_train.shape, np.min(y), np.max(y))

neurons = 200
drop = 0.1
act = 'relu'
inputs = layers.Input(shape=(1, ))
x = layers.Dense(neurons, activation=act)(inputs)
x = layers.Dense(neurons, activation=act)(x)
x = layers.Dense(neurons, activation=act)(x)
outputs = layers.Dense(1, activation='linear')(x)

model = models.Model(inputs=inputs, outputs=outputs)
# tf.contrib.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5)
opt = optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mean_absolute_percentage_error'])

try:
    history = model.fit(X_train, y_train, epochs=10, batch_size=50,
                        callbacks=[early_stop], validation_split=0.01)
except KeyboardInterrupt:
    pass
model.save("models/nn_dkn.h5")
# model = models.load_model("models/nn_dkn.h5")

b = time.time()
_, y_test = gaussian(rsc_X(X_test), 0, h)
a = time.time()
t_real = a - b

b = time.time()
y_pred = model.predict(X_test, batch_size=100)
a = time.time()
t_pred = a - b

print(np.dstack((sc_y(y_test.ravel()), y_pred.ravel())))
print(np.dstack((y_test.ravel(), rsc_y(y_pred.ravel()))))
print("Test sample: ", X_test.shape)
print("Real: {};  Pred: {}".format(t_real, t_pred))
