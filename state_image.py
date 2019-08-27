import numpy as np
import pandas as pd
from sklearn import preprocessing
from tools import check_dir
import matplotlib.pyplot as plt
import os
import h5py
import re
import csv


def find_max_nb(files, path=''):
    max_nb = 0
    for file in files:
        with open(path + file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                nbs = len(row)
                if nbs > max_nb:
                    max_nb = nbs
    return max_nb


class Scaler:
    def __init__(self, array):
        self.max_value = np.max(abs(array))

    def scale(self, array):
        return array / self.max_value

    def rescale(self, array):
        return array * self.max_value


data_path = "log/simulation/"
hdf_name = "../datafile"
data_files = sorted(os.listdir(data_path), key=lambda x: float(re.findall('\d+', x)[0]))
n_files = len(data_files)
N = pd.read_csv(data_path + data_files[-1]).shape[0]
pos = np.zeros((n_files, N, 2))
vel = np.zeros((n_files, N, 2))
acc = np.zeros((n_files, N, 2))
density = np.zeros((n_files, N, 1))
drho = np.zeros((n_files, N, 1))

print("Found {} files.".format(n_files))
print("Reading files...")
for i, file in enumerate(data_files):
    df = pd.read_csv(data_path + file)
    pos[i] = df[["xpos", "zpos"]].values
    vel[i] = df[["xvel", "zvel"]].values
    acc[i] = df[["xacc", "zacc"]].values
    density[i] = df["density"].values.reshape((-1,1))
    drho[i] = df["drho"].values.reshape((-1,1))

print("pos.shape: ", pos.shape)
print("vel.shape: ", vel.shape)
print("drho.shape: ", drho.shape)

# scx, scz = Scaler(pos[:, :, 0]), Scaler(pos[:, :, 1])
# scvx, scvz = Scaler(vel[:, :, 0]), Scaler(vel[:, :, 1])
# scrho = Scaler(drho)
# pos[:, :, 0], pos[:, :, 1] = scx.scale(pos[:, :, 0]), scz.scale(pos[:, :, 1])
# vel[:, :, 0], vel[:, :, 1] = scvx.scale(vel[:, :, 0]), scvz.scale(vel[:, :, 1])
# drho = scrho.scale(drho)

print("   Done!")
hf = h5py.File(data_path + hdf_name, mode="w")
hf.create_dataset('pos', data=pos)
hf.create_dataset('vel', data=vel)
hf.create_dataset('acc', data=acc)
hf.create_dataset('density', data=density)
hf.create_dataset('drho', data=drho)
hf.create_dataset('stats', data=(n_files, N))
# hf.create_dataset('scale', data=[scx.max_value, scz.max_value, scvx.max_value, scvz.max_value, scrho.max_value])
# print("Maximal values:", [scx.max_value, scz.max_value, scvx.max_value, scvz.max_value, scrho.max_value])
hf.close()
print("Dataset created as HDF5 file.")
