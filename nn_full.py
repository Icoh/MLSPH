import numpy as np
import scipy.linalg as lin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import tensorflow as tf
from tensorflow.contrib.keras import layers, models, callbacks, optimizers
import h5py
import os, time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def minmax(data):
    max_val = np.max(data)
    min_val = np.min(data)

    def scaler(array):
        print("Scaling with max={},  min={}".format(max_val, min_val))
        return (array - min_val)/ (max_val - min_val)

    def rescaler(array):
        print("Rescaling with max={},  min={}".format(max_val, min_val))
        return array * (max_val-min_val) + min_val
    return (data-min_val)/(max_val-min_val), scaler, rescaler


def gaussian(r, unit_vect, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g * (-unit_vect)
    return g, dg


def continuity(vdiff, dkernel, m=0.2):
    return m * np.sum(vdiff * dkernel, axis=-1)


h = 0.01103
support = 3
n_data = 1000000
n_nbs = 20
norms = np.random.rand(n_data, n_nbs, 1) * support * h
units = normalize((np.random.rand(n_data*n_nbs, 2) * np.random.choice([-1, 0, 1], (n_data*n_nbs, 2))))
units = units.reshape((n_data, n_nbs, 2))
posdiff = norms*units
veldiff = 0.05*(np.random.rand(n_data, n_nbs, 2) * np.random.choice([-1, 0, 1], (n_data, n_nbs, 2)))
kn, dkn = gaussian(norms, units, h)
drho = np.sum(continuity(veldiff, dkn), axis=-1)
print("norms", norms[:3])
print("units", units[:3])
print("posdiff", posdiff[:3])
print("veldiff", veldiff[:3])
print("drho", drho[:3])

mm_posdiff, sc_pos, resc_pos = minmax(posdiff)
mm_veldiff, sc_vel, resc_vel = minmax(veldiff)
mm_drho, sc_drho, resc_drho = minmax(drho)

vars = 4
print("Samples:", n_data*n_nbs)
print("Neighbor count:", n_nbs)
print("Preparing features and labels...")
X = np.zeros((n_nbs, n_data, vars))
X[:, :, 0] = mm_posdiff[:, :, 0].transpose()
X[:, :, 1] = mm_posdiff[:, :, 1].transpose()
X[:, :, 2] = mm_veldiff[:, :, 0].transpose()
X[:, :, 3] = mm_veldiff[:, :, 1].transpose()
X = list(X)
y = mm_drho

# Define Neural Network Model
act = 'relu'
inputs, hidden1, hidden2 = list(), list(), list()
for n in range(n_nbs):
    inputs.append(layers.Input(shape=(vars,)))
    hidden1.append(layers.Dense(2, activation=act)(inputs[n]))
x = layers.Add()(hidden1)
x = layers.Dense(10, activation=act)(x)
out = layers.Dense(1, activation='linear')(x)

model = models.Model(inputs=inputs, outputs=out)
tf.contrib.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='loss', patience=5)
opt = optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])
try:
    history = model.fit(X, y, epochs=10, batch_size=25,
                        callbacks=[early_stop], validation_split=0.01, shuffle=True)
except KeyboardInterrupt:
    pass

model.save("models/nn_full.h5")
# model = models.load_model("models/nn_full.h5")

samples = 10000
pdiff = resc_pos(np.random.rand(samples, 20, 2))
vdiff = resc_vel(np.random.rand(samples, 20, 2))

b = time.time()
r = lin.norm(pdiff, axis=-1).reshape((samples, 20, 1))
us = pdiff / r
_, dw = gaussian(r, us, h)
y_test = np.sum(continuity(vdiff, dw), axis=-1)
a = time.time()
t_real = a-b

X = np.zeros((20, samples, 4))
mm_pdiff = sc_pos(pdiff)
mm_vdiff = sc_vel(vdiff)
X[:, :, 0] = mm_pdiff[:, :, 0].transpose()
X[:, :, 1] = mm_pdiff[:, :, 1].transpose()
X[:, :, 2] = mm_vdiff[:, :, 0].transpose()
X[:, :, 3] = mm_vdiff[:, :, 1].transpose()
X = list(X)
b = time.time()
y_pred = resc_drho(model.predict(X))
a = time.time()
t_pred = a-b

print("Normal: {};   Pred: {}".format(t_real, t_pred))
print(np.dstack((y_test.ravel(),y_pred.ravel()))[:5])
y_test, y_pred = sc_drho(y_test).ravel(), sc_drho(y_pred).ravel()
print(np.dstack((y_test, y_pred))[:5])
