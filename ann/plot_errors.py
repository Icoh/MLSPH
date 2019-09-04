import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib.keras import models
from ann.equations import dgaussian, continuity
from sph.tools import check_dir

save_path = "../models/continuity/"
check_dir(save_path)

model1 = models.load_model(save_path + "h1/nn_250_0.h5")
model2 = models.load_model(save_path + "h2/nn_250_0.h5")
model3 = models.load_model(save_path + "h3/nn_250_0.h5")

X = np.linspace(0, 3, 250)
V = np.linspace(0, 1, 250)
X, V = np.meshgrid(X, V)
X = X.reshape((-1, 1))
V = V.reshape((-1, 1))
dkn = dgaussian(X, 1)
y = continuity(V, dkn).ravel()
y_pred1 = model1.predict(np.concatenate((X,V), axis=-1)).ravel()
y_pred2 = model2.predict(np.concatenate((X,V), axis=-1)).ravel()
y_pred3 = model3.predict(np.concatenate((X,V), axis=-1)).ravel()

plt.plot(X, y)
plt.plot(X, y_pred3)
plt.show()

plt.plot(y, abs(y - y_pred3.ravel()) / y, 'g.', label='3 capas')
plt.plot(y, abs(y-y_pred2.ravel())/y, 'c.', label='2 capas')
plt.plot(y, abs(y-y_pred1.ravel())/y, 'm.', label='1 capa')
plt.yscale('log')
plt.xscale('log')
plt.ylabel("Error absoluto relativo")
plt.xlabel(r"D(r, u)")
plt.legend()
plt.grid()
plt.show()
