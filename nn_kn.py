import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import scipy.spatial as sp
import scipy.linalg as lin
import matplotlib.pyplot as plt
import time
from tensorflow.contrib.keras import layers, models, callbacks, optimizers
import tensorflow as tf
import pickle


class Minmax:
    def __init__(self, data):
        self.data = data
        self.max_val = np.max(data)
        self.min_val = np.min(data)
        print("Scaler with max={},  min={}".format(self.max_val, self.min_val))

    def scale(self, data=None):
        if data is None:
            return (self.data - self.min_val) / (self.max_val - self.min_val)
        else:
            return (data - self.min_val) / (self.max_val - self.min_val)

    def rescale(self, data=None):
        if data is None:
            return self.data * (self.max_val-self.min_val) + self.min_val
        else:
            return data * (self.max_val-self.min_val) + self.min_val

    def save(self, path):
        with open('scalers/'+path, 'wb') as file:
            pickle.dump(self, file)


# def scale(data):
#     min_val = np.min(data)
#
#     def scaler(array):
#         print("Scaling with max={}".format(min_val))
#         return array-min_val
#
#     def rescaler(array):
#         print("Rescaling with max={}".format(min_val))
#         return array+min_val
#     return data-min_val, scaler, rescaler


def gaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    return g


def train(neurons, hidden=1, act='relu', epochs=10, rep=0):
    samples = int(1e6)
    h = 1
    norms = np.random.uniform(0, 3, (samples, 1))
    kn = gaussian(norms, h)

    X = norms
    y = kn

    inputs = layers.Input(shape=(1, ))
    x = layers.Dense(neurons, activation=act)(inputs)
    for i in range(hidden - 1):
        x = layers.Dense(neurons, activation=act)(x)
    outputs = layers.Dense(1, activation='linear')(x)

    save_path = "models/kernel/h{}/nn_{}_{}.h5".format(hidden, neurons, rep)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.summary()
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=5)
    check_point = callbacks.ModelCheckpoint(save_path,
                                            monitor='val_loss', save_best_only=True,
                                            mode='min')
    opt = optimizers.Adam(lr=1e-3, decay=1e-5)
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_percentage_error'])

    history = model.fit(X, y, epochs=epochs, batch_size=50,
                        callbacks=[check_point, early_stop], validation_split=0.01)
    return models.load_model(save_path)
