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


inputs_path = "log/inputs/"
targets_path = "log/targets/"
hdf_name = "../datafile"
inputs_files = sorted(os.listdir(inputs_path), key=lambda x: float(re.findall('\d+', x)[0]))
targets_files = sorted(os.listdir(inputs_path), key=lambda x: float(re.findall('\d+', x)[0]))
n_files = len(inputs_files)
N_in = pd.read_csv(inputs_path + inputs_files[-1]).shape[0]
N_out = pd.read_csv(targets_path + targets_files[-1]).shape[0]
pos = np.zeros((n_files, N_in, 2))
vel = np.zeros((n_files, N_in, 2))
acc = np.zeros((n_files, N_out, 2))
density = np.zeros((n_files, N_out, 1))
drho = np.zeros((n_files, N_out, 1))

print("Found {} files.".format(n_files))
print("Reading files...")
for i, file in enumerate(inputs_files):
    df_in = pd.read_csv(inputs_path + file)
    pos[i] = df_in[["xpos", "zpos"]].values
    vel[i] = df_in[["xvel", "zvel"]].values
for i, file in enumerate(targets_files):
    df_out = pd.read_csv(targets_path + file)
    acc[i] = df_out[["xacc", "zacc"]].values
    density[i] = df_out["density"].values.reshape((-1, 1))
    drho[i] = df_out["drho"].values.reshape((-1, 1))

print("pos.shape: ", pos.shape)
print("vel.shape: ", vel.shape)
print("drho.shape: ", drho.shape)

print("   Done!")
hf = h5py.File(inputs_path + hdf_name, mode="w")
hf.create_dataset('pos', data=pos)
hf.create_dataset('vel', data=vel)
hf.create_dataset('acc', data=acc)
hf.create_dataset('density', data=density)
hf.create_dataset('drho', data=drho)
hf.create_dataset('stats', data=(n_files, N_in, N_out))
# hf.create_dataset('scale', data=[scx.max_value, scz.max_value, scvx.max_value, scvz.max_value, scrho.max_value])
# print("Maximal values:", [scx.max_value, scz.max_value, scvx.max_value, scvz.max_value, scrho.max_value])
hf.close()
print("Dataset created as HDF5 file.")
