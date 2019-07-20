import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import os


data_dir = "log/"
files = os.listdir(data_dir)
files = sorted(files, key=lambda x: float(x.split("c")[1]))
print(files)
df = pd.read_csv(data_dir+files[0])
N, n_data = df.shape

print("Number of files: {}".format(len(files)))
print("Data with {} particles and {} values.".format(N, n_data))
print("Reading files...".format(files[0]), end='')
for path in files[1:]:
    dft = pd.read_csv(data_dir+path)
    df = df.append(dft)
print("   Done!")
print("Total data shape {}".format(df.shape))

X = df.drop(["xacc", "zacc"], axis=1).values
#X = X.reshape((len(files), N, 5))
X = X[:-1]
y = df[["xacc", "zacc"]].values
#y = y.reshape((len(files), N, 2))
y = y[1:]

# We will use 70% of the data for training and 30% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train)
print(y_train)

# Deep Learning model.
model = Sequential()
model.add(Dense(10, kernel_initializer="uniform", activation="relu", input_dim=n_data-2))
model.add(Dense(2, kernel_initializer="uniform", activation="linear"))
print(model.summary())

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=10, epochs=3)

y_pred = model.predict(X_test)
print(y_pred-y_test)
