import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from time import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os


def gaussian(r, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    return g


save_path = "./dnn/kernel/"
warm = False
if not warm:
    paths = os.listdir(save_path)
    for p in paths:
        os.remove(save_path+p)

samples = int(1e6)
X = np.random.uniform(0, 3, samples)
Y = gaussian(X, 1)

X, Y = shuffle(X, Y, random_state=0)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.01, random_state=0)


featcol = [tf.feature_column.numeric_column("dist")]
x = {'dist': train_x}
x_t = {'dist': test_x}
y = train_y

input_fn = tf.estimator.inputs.numpy_input_fn(x, y,
                                              num_epochs=5,
                                              shuffle=True,
                                              queue_capacity=1000,
                                              num_threads=4)

test_input_fn = tf.estimator.inputs.numpy_input_fn(x_t,
                                                   batch_size=100,
                                                   num_epochs=1,
                                                   shuffle=False)

config = tf.ConfigProto(device_count={'GPU': 1, 'CPU':4})
sess = tf.Session(config=config)

opti = tf.train.AdamOptimizer(learning_rate=0.001)
model = tf.estimator.DNNRegressor(feature_columns=featcol,
                                  hidden_units=[150, 150, 150],
                                  activation_fn=tf.nn.elu,
                                  optimizer=opti,
                                  model_dir=save_path)

model.train(input_fn, steps=10000)
start = time()
predictions = model.predict(test_input_fn)
end = time()
i = 0
pred_y = []
for pred in predictions:
    print(test_y[i], pred['predictions'][0])
    pred_y.append(pred['predictions'][0])
    i += 1

plt.plot(test_x, test_y, 'ro')
plt.plot(test_x, pred_y, 'k.')
plt.show()

print("Prediction time:", end - start)
print(np.sqrt(mean_squared_error(test_y, pred_y)) ** 0.5)
