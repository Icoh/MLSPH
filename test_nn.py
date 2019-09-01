import numpy as np
import matplotlib.pyplot as plt
import timeit
from tensorflow.contrib.keras import layers, models, callbacks, optimizers
import pickle
import csv
from nn_kn import train


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
            return self.data * (self.max_val - self.min_val) + self.min_val
        else:
            return data * (self.max_val - self.min_val) + self.min_val

    def save(self, path):
        with open('scalers/' + path, 'wb') as file:
            pickle.dump(self, file)


def load_scaler(path):
    with open('scalers/' + path, 'rb') as file:
        return pickle.load(file)


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
        subs = abs(real - prediction) / real
        mean = 100 * np.sum(subs) / subs.size
        return mean
    else:
        print("Sizes must match")
        return None


def gaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    return g


setup_code = '''
from tensorflow.contrib.keras import models
import numpy as np
samples = int(1e5)
X = np.random.uniform(0, 3, (samples, 1))
model = models.load_model("models/{}/h{}/nn_{}_{}.h5")
'''

title = 'kernel'
neurons = [10, 20, 50, 100, 200, 500]
repeat = 1
epochs= 1
act = 'relu'
hidden_units = 1

support = 3
samples = int(1e5)
print("Generating {} samples... ".format(samples))
norms = np.random.uniform(0, 3, (samples, 1))
kn = gaussian(norms, 1)
print("Done!")

# xmm, ymm = load_scaler(xsc_name), load_scaler(ysc_name)
X = norms
y = kn

rm_errors = ['rmse']
rm_std = ['std']
# ma_errors = ["mape"]
times = ["time"]

for n in neurons:
    n_errors = np.zeros(repeat)
    n_times = np.zeros(repeat)

    for i in range(repeat):
        print("  >> Repetition", i)
        print("   >> Training NN with {} neurons".format(n))
        model = train(n, hidden=hidden_units, act=act, epochs=epochs, rep=repeat)

        y_pred = model.predict(X)

        time = timeit.timeit(setup=setup_code.format(title, hidden_units, n, i),
                             stmt="model.predict(X)", number=1)
        print(np.dstack((y.ravel(), y_pred.ravel())))

        rm = rmse(y, y_pred)
        ma = mape(y, y_pred)
        print(">> Time:", time)
        print(">> RMSE:{};   MAPE:{}".format(rm, ma))
        n_errors[i] = rm
        n_times[i] = time

    times.append(np.mean(n_times))
    rm_errors.append(np.mean(n_errors))
    rm_std.append(np.std(n_errors, ddof=1))

log_file = open('models/{}/h{}/stats.csv'.format(title, hidden_units), 'w+')
writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow([""] + neurons)
writer.writerow(rm_errors)
writer.writerow(times)
log_file.close()
