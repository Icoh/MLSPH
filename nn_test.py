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


def continuity(vdiff, dkernel, m=0.2):
    return m * np.sum(vdiff * dkernel, axis=-1)


samples = [2 ** i for i in range(20)]
# config = tf.ConfigProto(device_count={'GPU': 0, 'CPU': 4})
# sess = tf.Session(config=config)

setup_end = '''
featcol = [tf.feature_column.numeric_column("dist")]
x = {'dist': X}
test_input_fn = tf.estimator.inputs.numpy_input_fn(x, batch_size=1,
                                                   num_epochs=1, shuffle=False)
opti = tf.train.AdamOptimizer(learning_rate=0.001)
model = tf.estimator.DNNRegressor(feature_columns=featcol, hidden_units=[100, 100, 100],
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

save_path = "./models/dnn/dkernel/"
samples = {}
X = np.random.uniform(0, 3, samples)'''

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

save_path = "./models/dnn/dkernel/"
X = np.linspace(0, 3, 100000)

featcol = [tf.feature_column.numeric_column("dist")]
x = {'dist': X}
test_input_fn = tf.estimator.inputs.numpy_input_fn(x, batch_size=100,
                                                   num_epochs=1, shuffle=False)
opti = tf.train.AdamOptimizer(learning_rate=0.001)
model = tf.estimator.DNNRegressor(feature_columns=featcol, hidden_units=[100, 100, 100],
                                  activation_fn=tf.nn.relu, optimizer=opti,
                                  model_dir=save_path)

y_real = dgaussian(X, 1)
predictions = model.predict(test_input_fn)
y_pred = [pred['predictions'][0] for pred in predictions]

rm = rmse(y_real, np.array(y_pred))
ma = mape(y_real, np.array(y_pred))
plt.plot(X, y_real, 'r-', label="Numpy")
plt.plot(X, y_pred, 'c--', label="RMSE {:.5f}\nMAPE {:.2f}".format(rm, ma))
plt.grid()
plt.legend()
plt.ylabel(r'$\nabla$ W(q)')
plt.xlabel('q')
plt.show()
