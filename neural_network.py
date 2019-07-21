import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
import os

data_dir = "./log_dam/"
files = os.listdir(data_dir)
files = sorted(files, key=lambda x: float(x.split("c")[1]))
print("Found {} files.".format(len(files)))
df = pd.read_csv(data_dir + files[0])
N, n_data = df.shape
print("Reading files...".format(files[0]), end='')
for path in files[1:]:
    dft = pd.read_csv(data_dir + path)
    df = df.append(dft)
print("   Done!")
print("Data from {} particles with {} values each.".format(N, n_data))
print("Total data shape {}".format(df.shape))

features = ["x", "xvel", "density"]
labels = "xacc"
features_df = df[features]
labels_df = df[labels]

X = features_df.values.reshape((len(files), N * 3))
y = labels_df.values.reshape((len(files), N))
X = X[1:]
y = y[:-1]
print("X shape:", X.shape)
print("y shape:", y.shape)

# We will use 70% of the data for training and 30% for testing.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

feats = pd.DataFrame(X_train, columns=[str(i) for i in range(X_train.shape[1])])
labels = pd.DataFrame(y_train, columns=[str(i) for i in range(y_train.shape[1])])
f_test = pd.DataFrame(X_test, columns=[str(i) for i in range(X_train.shape[1])])
l_test = pd.DataFrame(y_test, columns=[str(i) for i in range(y_train.shape[1])])


# Make Input Function
def input_fn():
    feat_dataset = tf.data.Dataset.from_tensor_slices(dict(feats))
    lab_dataset = tf.data.Dataset.from_tensor_slices(labels.values)
    dataset = tf.data.Dataset.zip((feat_dataset, lab_dataset))
    dataset = dataset.batch(1)
    return dataset


# Make Input Function
def eval_input_fn():
    feat_dataset = tf.data.Dataset.from_tensor_slices(dict(f_test))
    lab_dataset = tf.data.Dataset.from_tensor_slices(l_test.values)
    dataset = tf.data.Dataset.zip((feat_dataset, lab_dataset))
    dataset = dataset.batch(1)
    return dataset


# Make Feature Columns
feature_columns = []
for key in feats.keys():
    feature_columns.append(tf.feature_column.numeric_column(key=key))
# feature_columns = [tf.feature_column.numeric_column("x", shape=[N*3], dtype=tf.float32)]

MODEL_PATH = './DNNRegressors/'
hidden_layers = [16, 16]
dropout = 0.0

# Define DNN Regressor Model
model = tf.estimator.DNNRegressor(feature_columns=feature_columns,
                                  label_dimension=N,
                                  hidden_units=hidden_layers,
                                  optimizer='Adam',
                                  model_dir=MODEL_PATH)

# Train the DNN Regressor Estimator
r = model.train(input_fn=input_fn, steps=1000)
print("Training done!")
# Evaluate the Model
validation_metrics = {"MSE": tf.contrib.metrics.streaming_mean_squared_error}
metrics = model.evaluate(input_fn=eval_input_fn)
print(metrics)
