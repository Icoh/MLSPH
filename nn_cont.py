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

    def rescaler(array):
        print("Rescaling with max={},  min={}".format(max_val, min_val))
        return array * (max_val-min_val) + min_val
    return (data-min_val)/(max_val-min_val), rescaler


def scale(data):
    min_val = np.min(data)

    def rescaler(array):
        print("Rescaling with max={}".format(min_val))
        return array + min_val
    return data - min_val, rescaler


def continuity(vdiff, dkernel, m=0.11365):
    return m * np.sum(vdiff * dkernel, axis=-1)


def gaussian(r, unit_vect, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g *(-unit_vect)
    return g, dg


h = 0.009231
support = 3
samples = int(2e6)
print("samples: ", samples)
norms = np.random.rand(samples, 1) * support * h
units = normalize((np.random.rand(samples, 2) * np.random.choice([-1, 0, 1], (samples,2))))
posdiff = norms*units
veldiff = 0.01*(np.random.rand(samples, 2) * np.random.choice([-1, -1, 0, 1, 1], (samples,2)))
kn, dkn = gaussian(norms, units, h)
cont = continuity(veldiff, dkn).reshape(-1,1)
print("norms", norms[:3])
print("units", units[:3])
print("posdiff", posdiff[:3])
print("veldiff", veldiff[:3])
print("cont", cont[:3])

X = np.zeros((samples, 4))

X[:, 0:2], resc_pos = minmax(posdiff)
X[:, 2:4], resc_vel = minmax(veldiff)
y, resc_cont = scale(cont)
print(X[:3], np.min(X), np.max(X))
print(y[:3], np.min(y), np.max(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001)

neurons = 200
drop = 0.1
act = 'relu'
inputs = layers.Input(shape=(4, ))
x = layers.Dense(neurons, activation=act)(inputs)
x = layers.Dense(neurons, activation=act)(x)
x = layers.Dense(neurons, activation=act)(x)
outputs = layers.Dense(1, activation='linear')(x)

model = models.Model(inputs=inputs, outputs=outputs)
tf.contrib.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', patience=10)
opt = optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mean_absolute_percentage_error'])
try:
    history = model.fit(X_train, y_train, epochs=10, batch_size=50,
                        callbacks=[early_stop], validation_split=0.01)
except KeyboardInterrupt:
    pass

model.save("models/nn_cont.h5")
# model = models.load_model("models/nn_cont.h5")
ypred = model.predict(X_test)
ytest = y_test
comp = np.dstack((ytest.ravel(), ypred.ravel()))
print(comp)
print(resc_cont(comp))

sample = 100000
b = time.time()
pdiff = resc_pos(X_train[:sample, 0:2])
vdiff = resc_vel(X_train[:sample, 2:4])
_, dw = gaussian(pdiff, vdiff,h)
y_test = continuity(vdiff, dw)
a = time.time()
t_real = a-b

b = time.time()
y_pred = resc_cont(model.predict(X_train[:sample], batch_size=sample))
a = time.time()
t_pred = a-b

print("Real: {};  Pred: {}".format(t_real, t_pred))
print(np.dstack((y_test.ravel(), y_pred.ravel())))
