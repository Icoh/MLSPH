import numpy as np
import scipy.linalg as lin
import matplotlib.pyplot as plt
import timeit
import tensorflow as tf


def rmse(real, prediction):
    if real.size == prediction.size:
        subs = (real - prediction) ** 2
        mean = np.sum(subs) / subs.size
        return np.sqrt(mean)
    else:
        print("Sizes must match")
        return None


def mape(real, prediction):
    if real.size == prediction.size:
        valid = real != 0
        subs = abs(real - prediction)[valid] / real[valid]
        mean = 100 * np.sum(subs) / subs.size
        return mean
    else:
        print("Sizes must match")
        return None


def dgaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g
    return dg


def gaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    return g


def continuity(vdiff, dkernel):
    return (vdiff * dkernel).reshape(-1,1)


samples = [2 ** i for i in range(20)]
config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 4})
sess = tf.Session(config=config)

setup_end = '''
featcol = [tf.feature_column.numeric_column("dist"), tf.feature_column.numeric_column("vel")]
x = {'dist': X}
test_input_fn = tf.estimator.inputs.numpy_input_fn(x, batch_size=100,
                                                   num_epochs=1, shuffle=False)
opti = tf.train.AdamOptimizer(learning_rate=0.001)
model = tf.estimator.DNNRegressor(feature_columns=featcol, hidden_units=[250, 250, 250],
                                      activation_fn=tf.nn.relu, optimizer=opti,
                                      model_dir=save_path)'''

setup_min = '''
import numpy as np
import tensorflow as tf

def dgaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g
    return dg

def continuity(vdiff, dkernel):
    return (vdiff * dkernel).reshape(-1, 1)

save_path = "./models/dnn/continuity/"
samples = {}
norms = np.random.uniform(0, 3, samples)
veldiffs = np.random.uniform(0, 1, samples)

X = np.zeros((samples, 2))
X[:, 0] = norms
X[:, 1] = veldiffs'''

pred_times = []
real_times = []
for s in samples:
    pred_times.append(timeit.timeit("model.predict(test_input_fn)", setup=setup_min.format(s) + setup_end, number=5))
    real_times.append(timeit.timeit("dgaussian(X, 1)", setup=setup_min.format(s), number=5))

plt.plot(samples, real_times, 'mo', label="Numpy")
plt.plot(samples, pred_times, 'co', label="Red neuronal")
plt.yscale('log')
plt.xscale('log')
plt.legend()
plt.grid()
plt.xlabel("Muestras evaluadas")
plt.ylabel("Tiempo (s)")
plt.show()

save_path = "./models/dnn/continuity/"
norms = np.random.uniform(0, 3, 10000)
veldiffs = np.random.uniform(0, 1, 10000)
dkn = dgaussian(norms, 1)
cont = continuity(veldiffs, dkn)

X = np.zeros((10000, 2))
X[:, 0] = norms
X[:, 1] = veldiffs
y_real = cont

featcol = [tf.feature_column.numeric_column("dist"), tf.feature_column.numeric_column("vel")]
x = {'dist': norms, 'vel': veldiffs}
test_input_fn = tf.estimator.inputs.numpy_input_fn(x, batch_size=100,
                                                   num_epochs=1, shuffle=False)
opti = tf.train.AdamOptimizer(learning_rate=0.001)
model = tf.estimator.DNNRegressor(feature_columns=featcol, hidden_units=[250, 250, 250],
                                  activation_fn=tf.nn.relu, optimizer=opti,
                                  model_dir=save_path)

predictions = model.predict(test_input_fn)
y_pred = np.array([pred['predictions'][0] for pred in predictions])
print(y_pred.shape)
print(y_real.shape)

rm = rmse(y_real.ravel(), y_pred.ravel())
ma = mape(y_real.ravel(), y_pred.ravel())
plt.plot(X[:,1], y_real.ravel(), 'r.', label="Numpy")
plt.plot(X[:,1], y_pred.ravel(), 'c.', label="RMSE {:.5f}\nMAPE {:.2f}".format(rm, ma))
plt.grid()
plt.legend()
plt.ylabel(r'D(r,u)')
plt.xlabel('y')
plt.show()

plt.plot(X[:,0], y_real.ravel(), 'r.', label="Numpy")
plt.plot(X[:,0], y_pred.ravel(), 'c.', label="RMSE {:.5f}\nMAPE {:.2f}".format(rm, ma))
plt.grid()
plt.legend()
plt.ylabel(r'D(r,u)')
plt.xlabel('r')
plt.show()
