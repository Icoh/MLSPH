import numpy as np
import scipy.linalg as lin
from tensorflow.contrib.keras import models
import timeit
import tensorflow as tf
from equations import nnps


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


def dgaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = 2 * q / h * g
    return g, dg


def gaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    return g


def continuity(vdiff, dkernel, m=0.2):
    return m * np.sum(vdiff * dkernel, axis=-1)


samples = [10**i for i in range(8)]
config = tf.ConfigProto(device_count={'GPU': 1, 'CPU':4})
sess = tf.Session(config=config)
tf.keras.backend.set_session(sess)

setup_code = '''
import numpy as np
from tensorflow.contrib.keras import models

def gaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    return g

samples = {}
X = np.random.uniform(0, 3, (samples, 1))
model_path = "models/definitive/nn_kernel.h5"
model = models.load_model(model_path)'''

for s in samples:
    real_time = timeit.timeit("gaussian(X, 1)", setup=setup_code.format(s), number=2)
    pred_time = timeit.timeit("model.predict(X, batch_size=1000000)", setup=setup_code.format(s), number=2)

    print("Samples: ", s)
    print(real_time)
    print(pred_time)

# y_real, _ = gaussian(X, 1)
# y_pred = model.predict(X)

# plt.plot(X, y_real)
# plt.plot(X, y_pred, 'r.')
# plt.show()
