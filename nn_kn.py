import numpy as np
from tensorflow.contrib.keras import layers, models, callbacks, optimizers
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

    def rescale(self, data):
        return data * (self.max_val-self.min_val) + self.min_val

    def save(self, path):
        with open('scalers/'+path, 'wb') as file:
            pickle.dump(self, file)


def load_scaler(path):
    with open('scalers/' + path, 'rb') as file:
        return pickle.load(file)


def gaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    return g


def train(neurons, epochs=10, repetition=0, summary=False):
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

    save_path = "models/kernel/h{}/nn_{}_{}.h5".format(hidden, neurons, repetition)
    model = models.Model(inputs=inputs, outputs=outputs)
    early_stop = callbacks.EarlyStopping(monitor='val_mean_absolute_percentage_error', patience=10)
    check_point = callbacks.ModelCheckpoint(save_path,
                                            monitor='val_mean_absolute_percentage_error', save_best_only=True,
                                            mode='min')
    opt = optimizers.Adam(lr=1e-3, decay=1e-5)
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_percentage_error'])

    if summary:
        model.summary()
    history = model.fit(X, y, epochs=epochs, batch_size=50,
                        callbacks=[check_point, early_stop], validation_split=0.01)
    return models.load_model(save_path)
