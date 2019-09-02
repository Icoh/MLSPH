import numpy as np
import matplotlib.pyplot as plt
import timeit
import csv
import tensorflow as tf
from nn_cont import train

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)


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


def dgaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g
    return dg


def continuity(vdiff, dkernel):
    return (vdiff * dkernel).reshape(-1, 1)


title = 'continuity'
neurons = [10, 20, 50, 100, 150, 250]
repeat = 3
epochs = 5
act = 'relu'
hidden_units = [1, 2, 3]

samples = int(1e5)
print("Generating {} samples... ".format(samples))
norms = np.random.uniform(0, 3, samples)
veldiffs = np.random.uniform(0, 1, samples)
dkn = dgaussian(norms, 1)
cont = continuity(veldiffs, dkn)

X_test = np.zeros((samples, 2))
X_test[:, 0] = norms/3
X_test[:, 1] = veldiffs
y_test = cont
print("Done!")

for hid in hidden_units:
    rm_means = ['rmse']
    rm_std = ['rm std']
    ma_means = ["mape"]
    ma_std = ["ma std"]
    for n in neurons:
        n_rm = np.zeros(repeat)
        n_ma = np.zeros(repeat)
        n_times = np.zeros(repeat)

        for i in range(repeat):
            print("  >> Repetition", i)
            print("   >> Training NN with {} neurons".format(n))
            model = train(n, hidden=hid, act=act, epochs=epochs, repetition=i)

            y_pred = model.predict(X_test)

            print(np.dstack((y_test.ravel(), y_pred.ravel())))

            rm = rmse(y_test, y_pred)
            ma = mape(y_test, y_pred)
            print(">> RMSE:{};   MAPE:{}".format(rm, ma))
            n_rm[i] = rm
            n_ma[i] = ma

        rm_means.append(np.mean(n_rm))
        ma_means.append(np.mean(n_ma))
        if repeat > 1:
            rm_std.append(np.std(n_rm, ddof=1))
            ma_std.append(np.std(n_ma, ddof=1))
            print(" >>>>>>>>>>>>> END OF REPETITIONS <<<<<<<<<<<<")
            print(">> RMSE:{};   MAPE:{}".format(rm_means[-1], ma_means[-1]))
            print(">> std:{},    std:{}".format(rm_std[-1], ma_std[-1]))

    log_file = open('models/{}/h{}/stats.csv'.format(title, hid), 'w+')
    writer = csv.writer(log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow([""] + neurons)
    writer.writerow(rm_means)
    writer.writerow(rm_std)
    writer.writerow(ma_means)
    writer.writerow(ma_std)
    log_file.close()
