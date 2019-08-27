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


def split_train_test(data, targets, test):
    samples = targets.shape[0]
    split = int(samples * test)
    x_tr, x_ts = data[:, split:], data[:, :split]
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

data_path = "log/datafile_10"
print("Reading Dataset...")
hf = h5py.File(data_path, mode='r+')

pos = np.array(hf.get('pos'))
vel = np.array(hf.get('vel'))
rho = np.array(hf.get('density'))
acc = np.array(hf.get('acc'))
drho = np.array(hf.get('drho'))
n_files, N = np.array(hf.get('stats'))

dkn = np.array(hf.get('dkn'))
if None in dkn:
    print(' -Calculating dkernels...')
    dkn = np.zeros((n_files, N, 2))
    for i, x in enumerate(pos):
        nnbs = nnps(support, h, x[:, 0], x[:, 1])
        dkn[i] = calculate_kernel(h, N, x[:, 0], x[:, 1], nnbs)
    hf.create_dataset('dkn', data=dkn)
print('Done!')
hf.close()

print("pos", pos[:3])
print("vel", vel[:3])
print("drho", drho[:3])

mm_pos, sc_pos, resc_pos = minmax(pos)
mm_vel, sc_vel, resc_vel = minmax(vel)
mm_drho, sc_drho, resc_drho = minmax(drho)
mm_dkn, sc_dkn, resc_dkn = minmax(dkn)

print("Samples:", n_files)
print("Number of particles:", N)
print("Preparing features and labels...")
vars = 2
X = np.zeros((N, n_files, vars))
V = np.zeros((N, n_files, vars))
y_dkn = np.zeros((n_files, 2 * N))
y_drho = np.zeros((n_files, N))
X[:, :, 0] = mm_pos[:, :, 0].transpose()
X[:, :, 1] = mm_pos[:, :, 1].transpose()
V[:, :, 0] = mm_vel[:, :, 0].transpose()
V[:, :, 1] = mm_vel[:, :, 1].transpose()
X = np.concatenate((X, V), axis=0)
y_dkn[:, :N] = mm_dkn[:, :, 0]
y_dkn[:, N:] = mm_dkn[:, :, 1]
y_drho[:, :N] = mm_drho[:, :, 0]

X_train, X_test, y_train_dkn, y_test_dkn = split_train_test(X, y_dkn, 0.01)
_, _, y_train_drho, y_test_drho = split_train_test(X, y_drho, 0.01)
print("X train shape:", X_train.shape)
print("y train shape:", y_train_dkn.shape)
print("X test shape:", X_test.shape)
print("y test shape:", y_test_dkn.shape)
X_train, X_test = list(X_train), list(X_test)

# Define Neural Network Model
act = 'relu'

inputs_pos, hidden = list(), list()
for i in range(N):
    inputs_pos.append(layers.Input(shape=(2,)))
    # hidden.append(layers.Dense(1, activation=act)(inputs[i]))
x = layers.concatenate(inputs_pos)
x = layers.Dense(8 * N, activation=act)(x)
x = layers.Dense(8 * N, activation=act)(x)
# x = layers.Dense(4*N, activation=act)(x)
out_pos = layers.Dense(2 * N, activation='linear')(x)

inputs_vel = list()
for i in range(N):
    inputs_vel.append(layers.Input(shape=(2,)))
z_drho = layers.concatenate(inputs_vel)
z_drho = layers.Dot(axes=-1)([out_pos, z_drho])
z_drho = layers.Dense(4 * N, activation=act)(z_drho)
z_drho = layers.Dense(4 * N, activation=act)(z_drho)
out_vel = layers.Dense(N, activation='linear')(z_drho)

model = models.Model(inputs=[inputs_pos, inputs_vel], outputs=[out_pos, out_vel])
# tf.contrib.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)
opt = optimizers.Adam(lr=1e-3, decay=1e-5)
model.compile(optimizer=opt,
              loss=['mean_squared_error', 'mean_squared_error'],
              loss_weights=[1.0, 1.0]
              # ,metrics=['mean_absolute_error', 'mean_squared_error']
              )
try:
    history = model.fit(X_train, [y_train_dkn, y_train_drho], epochs=100, batch_size=5,
                        callbacks=[early_stop], validation_split=0.01, shuffle=True)
except KeyboardInterrupt:
    pass

model.save("models/nn_full.h5")

# b = time.time()
# r = lin.norm(pdiff, axis=-1).reshape((samples, 20, 1))
# us = pdiff / r
# _, dw = gaussian(r, us, h)
# y_test = np.sum(continuity(vdiff, dw), axis=-1)
# a = time.time()
# t_real = a-b

b = time.time()
y_pred_dkn, y_pred_drho = model.predict(X_test)
y_pred_dkn, y_pred_drho = resc_dkn(y_pred_dkn), resc_drho(y_pred_drho)
a = time.time()
t_pred = a - b

print("Pred: {}s".format(t_pred))
print("dkernels")
print(np.dstack((y_test_dkn.ravel(), sc_dkn(y_pred_dkn.ravel()))))
print(np.dstack((resc_dkn(y_test_dkn.ravel()), y_pred_dkn.ravel())))
print('drhos')
print(np.dstack((y_test_drho.ravel(), sc_dkn(y_pred_drho.ravel()))))
print(np.dstack((resc_dkn(y_test_drho.ravel()), y_pred_drho.ravel())))

errs = (abs(sc_dkn(y_test_dkn.ravel()) - y_pred_dkn.ravel()))
plt.plot(resc_dkn(y_test_dkn.ravel()), errs, 'k.')
plt.show()
