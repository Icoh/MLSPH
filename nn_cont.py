import numpy as np
from tensorflow.contrib.keras import layers, models, callbacks, optimizers


def continuity(vdiff, dkernel):
    return (vdiff * dkernel).reshape(-1, 1)


def gaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g
    return dg


def train(neurons, hidden=1, act='relu', epochs=10, repetition=0):
    samples = int(1e6)
    norms = np.random.uniform(0, 3, (samples, 1))
    veldiffs = np.random.uniform(0, 0.01, (samples, 1))
    dkn = gaussian(norms, 1)
    cont = continuity(veldiffs, dkn)

    X = np.zeros((samples, 2))
    X[:, 0] = norms
    X[:, 1] = veldiffs
    y = cont

    inputs = layers.Input(shape=(2,))
    x = layers.Dense(neurons, activation=act)(inputs)
    for i in range(hidden - 1):
        x = layers.Dense(neurons, activation=act)(x)
    outputs = layers.Dense(1, activation='linear')(x)

    save_path = "models/continuity/h{}/nn_{}_{}.h5".format(hidden, neurons, repetition)
    model = models.Model(inputs=inputs, outputs=outputs)
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)
    check_point = callbacks.ModelCheckpoint(save_path,
                                            monitor='val_loss', save_best_only=True,
                                            mode='min')
    opt = optimizers.Adam(lr=1e-3, decay=1e-5)
    model.compile(optimizer=opt,
                  loss='mean_squared_error',
                  metrics=['mean_absolute_percentage_error'])

    history = model.fit(X, y, epochs=5, batch_size=100,
                        callbacks=[early_stop, check_point], validation_split=0.01)
    return models.load_model(save_path)
