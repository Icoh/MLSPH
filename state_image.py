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


def run(plot=False, *args, **kwags):
    xpos_path = "cnn_files/xdiff/"
    zpos_path = "cnn_files/xdiff/"
    xvel_path = "cnn_files/xvdiff/"
    zvel_path = "cnn_files/zvdiff/"
    drho_path = "cnn_files/drho/"
    data_path = "cnn_files/cnn_data/"
    hdf_name = "datafile"
    xpos_files = os.listdir(xpos_path)
    zpos_files = os.listdir(zpos_path)
    xvel_files = os.listdir(xvel_path)
    zvel_files = os.listdir(zvel_path)
    drho_files = os.listdir(drho_path)
    xpos_files = sorted(xpos_files, key=lambda x: float(re.findall('\d+', x)[0]))
    zpos_files = sorted(zpos_files, key=lambda x: float(re.findall('\d+', x)[0]))
    xvel_files = sorted(xvel_files, key=lambda x: float(re.findall('\d+', x)[0]))
    zvel_files = sorted(zvel_files, key=lambda x: float(re.findall('\d+', x)[0]))
    drho_files = sorted(drho_files, key=lambda x: float(re.findall('\d+', x)[0]))
    n_files = len(xpos_files)

    print("Found {} files.".format(n_files))
    print("Reading files...")
    mnb = find_max_nb(xpos_files, path=xpos_path)
    xpos0 = pd.read_csv(xpos_path + xpos_files[0], names=range(mnb)).fillna(0)
    n_particles, n_nbs = xpos0.shape
    pos = np.zeros((n_files * n_particles, n_nbs, 2))
    vel = np.zeros((n_files * n_particles, n_nbs, 2))
    drho = np.zeros((n_files * n_particles, 1))

    print("pos.shape: ", pos.shape)
    print("vel.shape: ", vel.shape)
    print("drho.shape: ", drho.shape)

    print("drho")
    for i, path in enumerate(drho_files):
        if not i % 100:
            print(i, "/", n_files)
        drho[i * n_particles:(i + 1) * n_particles] = pd.read_csv(drho_path + path, names=[0])
    print("pos")
    for i, (xpath, zpath) in enumerate(zip(xpos_files, zpos_files)):
        if not i % 100:
            print(i, "/", n_files)
        pos[i * n_particles:(i + 1) * n_particles, :, 0] = pd.read_csv(xpos_path + xpath, names=range(mnb)).fillna(0)
        pos[i * n_particles:(i + 1) * n_particles, :, 1] = pd.read_csv(zpos_path + zpath, names=range(mnb)).fillna(0)
    print("vel")
    for i, (xpath, zpath) in enumerate(zip(xvel_files, zvel_files)):
        if not i % 100:
            print(i, "/", n_files)
        vel[i * n_particles:(i + 1) * n_particles, :, 0] = pd.read_csv(xvel_path + xpath, names=range(mnb)).fillna(0)
        vel[i * n_particles:(i + 1) * n_particles, :, 1] = pd.read_csv(zvel_path + zpath, names=range(mnb)).fillna(0)

    # scx, scz = Scaler(pos[:, :, 0]), Scaler(pos[:, :, 1])
    # scvx, scvz = Scaler(vel[:, :, 0]), Scaler(vel[:, :, 1])
    # scrho = Scaler(drho)
    # pos[:, :, 0], pos[:, :, 1] = scx.scale(pos[:, :, 0]), scz.scale(pos[:, :, 1])
    # vel[:, :, 0], vel[:, :, 1] = scvx.scale(vel[:, :, 0]), scvz.scale(vel[:, :, 1])
    # drho = scrho.scale(drho)

    print("   Done!")
    hf = h5py.File(data_path + hdf_name, mode="w")
    hf.create_dataset('posdiff', data=pos)
    hf.create_dataset('veldiff', data=vel)
    hf.create_dataset('drho', data=drho)
    hf.create_dataset('shape', data=pos.shape)
    # hf.create_dataset('scale', data=[scx.max_value, scz.max_value, scvx.max_value, scvz.max_value, scrho.max_value])
    # print("Maximal values:", [scx.max_value, scz.max_value, scvx.max_value, scvz.max_value, scrho.max_value])
    hf.close()
    print("Dataset created as HDF5 file.")
    return
