import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt
import os
from ann.equations import gaussian, dgaussian, continuity, rmse, mape
from sph.tools import check_dir

save_path = "../models/dnn/continuity/"
check_dir("../models")
check_dir("../models/dnn")
check_dir(save_path)
warm = False
if not warm:
    paths = os.listdir(save_path)
    for p in paths:
        os.remove(save_path + p)

samples = int(1e6)
norms = np.random.uniform(0, 3, samples)
veldiffs = np.random.uniform(0, 1, samples)
dkn = dgaussian(norms, 1)
cont = continuity(veldiffs, dkn)

X = np.zeros((samples, 2))
X[:, 0] = norms
X[:, 1] = veldiffs
Y = cont
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.01, random_state=0)
print("Continuity range:", np.min(Y), np.max(Y))

featcol = [tf.feature_column.numeric_column("dist"), tf.feature_column.numeric_column("vel")]
x = {'dist': train_x[:, 0], 'vel': train_x[:, 1]}
x_t = {'dist': test_x[:, 0], 'vel': test_x[:, 1]}
y = train_y

input_fn = tf.estimator.inputs.numpy_input_fn(x, y, num_epochs=20, shuffle=True,
                                              queue_capacity=1000, num_threads=1)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x_t, batch_size=100,
                                                   num_epochs=1, shuffle=False)

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU': 4})
sess = tf.Session(config=config)

opti = tf.train.AdamOptimizer(learning_rate=0.001)
model = tf.estimator.DNNRegressor(feature_columns=featcol, hidden_units=[250, 250, 250],
                                  activation_fn=tf.nn.relu, optimizer=opti,
                                  model_dir=save_path)

model.train(input_fn, steps=100000)


def get_pred(data):
    return data['predictions'][0]


start = time()
pred_y = list(map(get_pred, model.predict(test_input_fn)))
end = time()
pred_time = end - start

plt.plot(test_x[:, 0], test_y, 'm.')
plt.plot(test_x[:, 0], pred_y, 'g.')
plt.show()
plt.plot(test_x[:, 1], test_y, 'm.')
plt.plot(test_x[:, 1], pred_y, 'g.')
plt.show()

print("Test samples:", test_x.shape)
print("Prediction time:", pred_time)
print(rmse(test_y.ravel(), np.array(pred_y).ravel()))
print(mape(test_y.ravel(), np.array(pred_y).ravel()))
