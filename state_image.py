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
        with open(path+file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for row in csv_reader:
                nbs = len(row)
                if nbs > max_nb:
                    max_nb = nbs
    return max_nb


def run(plot=False, *args, **kwags):
    pos_path = "cnn_files/posdiff/"
    vel_path = "cnn_files/veldiff/"
    drho_path = "cnn_files/drho/"
    data_path = "cnn_files/cnn_data/"
    hdf_name = "datafile"
    pos_files = os.listdir(pos_path)
    vel_files = os.listdir(vel_path)
    drho_files = os.listdir(drho_path)
    pos_files = sorted(pos_files, key=lambda x: float(re.findall('\d+', x)[0]))
    vel_files = sorted(vel_files, key=lambda x: float(re.findall('\d+', x)[0]))
    drho_files = sorted(drho_files, key=lambda x: float(re.findall('\d+', x)[0]))
    n_files = len(pos_files)

    print("Found {} files.".format(n_files))
    print("Reading files...")
    mnb = find_max_nb(pos_files, path=pos_path)
    pos0 = pd.read_csv(pos_path + pos_files[0], names=range(mnb)).fillna(0)
    vel0 = pd.read_csv(vel_path + vel_files[0], names=range(mnb)).fillna(0)
    n_particles, n_nbs = pos0.shape
    pos = np.zeros((n_files*n_particles, n_nbs))
    vel = np.zeros((n_files*n_particles, n_nbs))
    drho = np.zeros((n_files*n_particles, 1))
    pos[:n_particles], vel[:n_particles] = pos0, vel0

    print("pos.shape: ", pos.shape)
    print("vel.shape: ", vel.shape)
    print("drho.shape: ", drho.shape)

    print("drho")
    for i, path in enumerate(drho_files):
        if not i % 100:
            print(i, "/", n_files)
        drho[i*n_particles:(i+1)*n_particles] = pd.read_csv(drho_path + path, names=[0])
    print("pos")
    for i, path in enumerate(pos_files[1:], 1):
        if not i % 100:
            print(i, "/", n_files)
        pos[i*n_particles:(i+1)*n_particles] = pd.read_csv(pos_path + path, names=range(mnb)).fillna(0)
    print("vel")
    for i, path in enumerate(vel_files[1:], 1):
        if not i % 100:
            print(i, "/", n_files)
        vel[i*n_particles:(i+1)*n_particles] = pd.read_csv(vel_path + path, names=range(mnb)).fillna(0)

    print("   Done!")
    hf = h5py.File(data_path+hdf_name, mode="w")
    hf.create_dataset('posdiff', data=pos)
    hf.create_dataset('veldiff', data=vel)
    hf.create_dataset('drho', data=drho)
    hf.create_dataset('shape', data=pos.shape)
    hf.close()
    print("Dataset created as HDF5 file.")
    return
