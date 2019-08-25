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
        return array * (max_val - min_val) + min_val

    return (data - min_val) / (max_val - min_val), rescaler


def scale(data):
    max_val = np.max(abs(data))

    def rescaler(array):
        print("Rescaling with max={}".format(max_val))
        return array * max_val

    return data / max_val, rescaler


def eos_tait(c, rho):
    gamma = 7.
    rho0 = 1000.
    b = c ** 2 * rho0 / gamma
    p = b * ((rho / rho0) ** gamma - 1)
    return p


def pressure_term(rhoa, pressa, rhob, pressb, dkernel, mass=0.2):
    c = (pressa / rhoa ** 2 + pressb / rhob ** 2).reshape(-1, 1)
    return mass * c * dkernel #* -1


def gaussian(r, unit_vect, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g  # * (-unit_vect)
    return g, dg


h = 0.01103
support = 3
samples = 10000000
print("samples: ", samples)
norms = np.random.rand(samples, 1) * support * h
units = normalize((np.random.rand(samples, 2) * np.random.choice([-1, 0, 1], (samples, 2))))
posdiff = norms * units
veldiff = 0.05 * (np.random.rand(samples, 2) * np.random.choice([-1, 0, 1], (samples, 2)))
kn, dkn = gaussian(norms, units, h)
a_dens = 1000 + np.random.rand(samples, 1) * 50
b_dens = np.random.permutation(a_dens)
a_pres = eos_tait(30, a_dens)
b_pres = eos_tait(30, b_dens)
acc = pressure_term(a_dens, a_pres, b_dens, b_pres, dkn).reshape(-1, 1)
# print("norms", norms[:3])
# print("units", units[:3])
# print("posdiff", posdiff[:3])
# print("veldiff", veldiff[:3])
# print("cont", acc[:3])

X = np.zeros((samples, 6))

X[:, 0:2], resc_pos = minmax(posdiff)
X[:, 2:4], resc_vel = minmax(veldiff)
X[:, 4:6], resc_pres = minmax(np.concatenate((a_dens, b_dens), axis=-1))
y, resc_acc = minmax(acc)
print(X[:3], np.min(X), np.max(X))
print(y[:3], np.min(y), np.max(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01)

neurons = 10
drop = 0.1
act = 'elu'
inputs = layers.Input(shape=(6,))
x = layers.Dense(neurons, activation=act)(inputs)
x = layers.Dense(neurons, activation=act)(x)
x = layers.Dense(neurons, activation=act)(x)
x = layers.Dense(neurons, activation=act)(x)
x = layers.Dense(neurons, activation=act)(x)
outputs = layers.Dense(1, activation='linear')(x)

model = models.Model(inputs=inputs, outputs=outputs)
tf.contrib.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', patience=5)
opt = optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
try:
    history = model.fit(X_train, y_train, epochs=10, batch_size=50,
                        callbacks=[early_stop], validation_split=0.01)
except KeyboardInterrupt:
    pass

model.save("models/nn_cont.h5")
# model = models.load_model("models/nn_cont.h5")

pdiff = resc_pos(X_test[:, 0:2])
vdiff = resc_vel(X_test[:, 2:4])
dens = 1000 + np.random.rand(X_test.shape[0], 2) * 50
pres = eos_tait(30, dens)
b = time.time()
_, dw = gaussian(pdiff, vdiff, h)
acc = pressure_term(dens[:, 0], pres[:, 0], dens[:, 1], pres[:, 1], dw).reshape(-1, 1)
a = time.time()
t_real = a - b

b = time.time()
y_pred = resc_acc(model.predict(X_test))
a = time.time()
t_pred = a - b

print("Real: {};  Pred: {}".format(t_real, t_pred))
print(np.dstack((resc_acc(y_test.ravel()), y_pred.ravel())))
