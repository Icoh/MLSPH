import numpy as np
import scipy.linalg as lin
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.keras import layers, models, callbacks, optimizers
import h5py
import os, time
from equations import nnps

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train_test_split(data, targets, test_size):
    samples = targets.shape[0]
    split = int(samples * test_size)
    x_tr, x_ts = data[split:], data[:split]
    y_tr, y_ts = targets[split:], targets[:split]
    return x_tr, x_ts, y_tr, y_ts


def minmax(data):
    max_val = np.max(data)
    min_val = np.min(data)

    def scaler(array):
        print("Scaling with max={},  min={}".format(max_val, min_val))
        return (array - min_val) / (max_val - min_val)

    def rescaler(array):
        print("Rescaling with max={},  min={}".format(max_val, min_val))
        return array * (max_val - min_val) + min_val
    return (data - min_val) / (max_val - min_val), scaler, rescaler


def scale(data):
    min_val = np.min(data)

    def scaler(array):
        print("Scaling with min={}".format(min_val))
        return array - min_val

    def rescaler(array):
        print("Rescaling with max={}".format(min_val))
        return array + min_val
    return data - min_val, scaler, rescaler


def calculate_kernel(h, N, x0, z0, nn_list):
    dkernels = np.zeros((N, 2), dtype=np.float64)

    for i, nbs in enumerate(nn_list):
        i_x, i_z = x0[i], z0[i]
        j_x, j_z = x0[nbs], z0[nbs]

        posdiff = np.dstack((i_x - j_x, i_z - j_z)).reshape(-1, 2)
        r = lin.norm(posdiff, axis=-1).reshape(-1, 1)
        posunit = posdiff / r

        _, dkn = gaussian(r, posunit, h)
        dkernels[i] = sum(dkn)
    return dkernels


def gaussian(r, unit_vect, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g * (-unit_vect)
    return g, dg


def continuity(vdiff, dkernel, m=0.2):
    return m * np.sum(vdiff * dkernel, axis=-1)


h = 0.01103
support = 3

data_path = "log/datafile"
print("Reading Dataset...")
hf = h5py.File(data_path, mode='r+')

sample = 2000
pos = np.array(hf.get('pos'))[:sample]
vel = np.array(hf.get('vel'))[:sample]
rho = np.array(hf.get('density'))[:sample]
acc = np.array(hf.get('acc'))[:sample]
drho = np.array(hf.get('drho'))[:sample]
n_files, N_in, N_out = np.array(hf.get('stats'))

dkn = np.array(hf.get('dkn'))[:sample]
if None in dkn:
    print(' -Calculating dkernels...')
    dkn = np.zeros((n_files, N_out, 2))
    for i, x in enumerate(pos):
        nnbs = nnps(support, h, x[:, 0], x[:, 1])
        dkn[i] = calculate_kernel(h, N_out, x[:, 0], x[:, 1], nnbs[:N_out])
    hf.create_dataset('dkn', data=dkn)
print('Done!')
hf.close()

print("pos", pos[:3])
print("vel", vel[:3])
print("drho", drho[:3])

mm_pos, sc_pos, resc_pos = minmax(pos)
mm_vel, sc_vel, resc_vel = minmax(vel)
mm_drho, sc_drho, resc_drho = scale(drho)

print("Samples:", sample)
print("Number of input particles:", N_in)
print("Number of output particles:", N_out)
print("Preparing features and labels...")
vars = 4
X = np.zeros((sample, N_in, vars))
y_drho = np.zeros((sample, N_out))
X[:, :, 0] = mm_pos[:, :, 0]
X[:, :, 1] = mm_pos[:, :, 1]
X[:, :, 2] = mm_vel[:, :, 0]
X[:, :, 3] = mm_vel[:, :, 1]

y_drho[:, :N_out] = mm_drho[:, :, 0]

X_train, X_test, y_train_drho, y_test_drho = train_test_split(X, y_drho, test_size=0.01)
print("X train shape:", X_train.shape)
print("y train shape:", y_train_drho.shape)
print("X test shape:", X_test.shape)
print("y test shape:", y_test_drho.shape)

# Define Neural Network Model
act = 'relu'

inputs = layers.Input(shape=(N_in, vars))
# x = layers.Conv1D(N_out, 20, activation=act)(inputs)
# x = layers.Conv1D(N_out, 20, activation=act)(x)
# x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(N_out, 20, activation=act)(inputs)
x = layers.Conv1D(N_out, 20, activation=act)(x)
x = layers.GlobalAveragePooling1D()(x)
# x = layers.Dropout(0.5)(x)
x = layers.Dense(10*N_out, activation='relu')(x)
x = layers.Dense(5*N_out, activation='relu')(x)
out = layers.Dense(N_out, activation='linear')(x)


model = models.Model(inputs=inputs, outputs=out)
# tf.contrib.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)
opt = optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mean_absolute_percentage_error'])
try:
    history = model.fit(X_train, y_train_drho[:, :N_out], epochs=100, batch_size=10,
                        callbacks=[early_stop], validation_split=0.01, shuffle=True)
except KeyboardInterrupt:
    pass

model.save("models/nn_full.h5")

b = time.time()
pdiff = resc_pos(X_test[:,:, 0:2])
vdiff = resc_vel(X_test[:,:, 2:4])
r = lin.norm(pdiff, axis=-1).reshape((-1, N_in, 1))
us = pdiff / r
_, dw = gaussian(r, us, h)
y_test = np.sum(continuity(vdiff, dw), axis=-1)
a = time.time()
t_real = a-b

b = time.time()
y_pred_dkn = resc_drho(model.predict(X_test))
a = time.time()
t_pred = a - b

print("Pred: {}s".format(t_pred))
print("dkernels")
print(np.dstack((y_test_drho.ravel(), sc_drho(y_pred_dkn.ravel()))))
print(np.dstack((resc_drho(y_test_drho.ravel()), y_pred_dkn.ravel())))
# print('drhos')
# print(np.dstack((y_test_drho.ravel(), sc_dkn(y_pred_drho.ravel()))))
# print(np.dstack((resc_dkn(y_test_drho.ravel()), y_pred_drho.ravel())))

errs = (abs(sc_drho(y_test_drho.ravel()) - y_pred_dkn.ravel()))
plt.plot(resc_drho(y_test_drho.ravel()), errs, 'k.')
plt.show()
