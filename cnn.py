import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.contrib.keras import layers, models, callbacks
import state_image as sti
import os
import h5py


data_dir = "./cnn_files/cnn_data/"
files = os.listdir(data_dir)
hdf_name = "datafile"
if hdf_name not in files:
    print("Executing state_image.py first!")
    sti.run(plot=False, skip=100)
else:
    print('Found', hdf_name, 'file.')
print("Starting CNN...")

hf = h5py.File(data_dir + hdf_name, 'r')
pos = np.array(hf.get('posdiff'))
vel = np.array(hf.get('veldiff'))
drho = np.array(hf.get('drho'))
hf.close()

# Use 90% of data for training
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
# print("Reshaped features and labels.")
# print("X_train shape:", X_train.shape)
# print("y_train shape:", y_train.shape)

n_data, n_nbs = pos.shape

# Define Neural Network Model
in_pos = layers.Input(shape=(n_nbs,))
x = layers.Dense(128, activation='relu', name='pos1')(in_pos)
x = layers.Dense(256, activation='relu', name='pos2')(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu', name='pos3')(x)

in_vel = layers.Input(shape=(n_nbs,))
v = layers.Dense(128, activation='relu', name='vel1')(in_vel)
v = layers.Dense(256, activation='relu', name='vel2')(v)
v = layers.BatchNormalization()(v)
v = layers.Dense(256, activation='relu', name='vel3')(v)

xv = layers.Concatenate()([x, v])
xv = layers.Dense(256, activation='relu', name='merge1')(xv)
xv = layers.BatchNormalization()(xv)
xv = layers.Dense(128, activation='relu', name='merge2')(xv)
out = layers.Dense(1, activation='linear')(xv)

model = models.Model(inputs=[in_pos, in_vel], outputs=out)
tf.contrib.keras.utils.plot_model(model, to_file='multilayer_perceptron_graph.png')
model.summary()
early_stop = callbacks.EarlyStopping(monitor='mean_absolute_error', patience=5)
model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

history = model.fit([pos, vel], drho, epochs=50, batch_size=128,
                    callbacks=[early_stop])

index = -100
model.evaluate([[pos[index]], [vel[index]]], [drho[index]])
model.save("model.h5")
