import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
import os
import pandas as pd
import h5py
from state_image import find_max_nb


def kernel(r, unit_vect, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = -2 * q / h * g * unit_vect
    return g, dg


def continuity(m, vdiff, dkernel):
    return m * np.sum(vdiff * dkernel, axis=-1)


def calculate_continuity(dists, unit_vects, veldiffs, m):
    _, dkn = kernel(dists, unit_vects, h)
    cont = continuity(m, veldiffs, dkn)
    return cont, dkn


h = 0.00750
support = 4

xpos_path = "cnn_files/xdiff/"
zpos_path = "cnn_files/xdiff/"
xvel_path = "cnn_files/xvdiff/"
zvel_path = "cnn_files/zvdiff/"

xpos_files = os.listdir(xpos_path)
zpos_files = os.listdir(zpos_path)
xvel_files = os.listdir(xvel_path)
zvel_files = os.listdir(zvel_path)
n_files = len(xpos_files)

print("Found {} files.".format(n_files))
print("Reading files...")
mnb = find_max_nb(xpos_files, path=xpos_path)
xpos0 = pd.read_csv(xpos_path + xpos_files[0], names=range(mnb)).fillna(0)
n_particles, n_nbs = xpos0.shape
pos = np.zeros((n_files * n_particles, n_nbs, 2))
vel = np.zeros((n_files * n_particles, n_nbs, 2))

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

r = lin.norm(pos, axis=-1)
r = r.reshape(*r.shape, 1)
units = pos / r
units[np.isnan(units)] = np.zeros_like(units[np.isnan(units)])
mass = 0.1497 * np.ones((n_files*n_particles, n_nbs))
conts, dkernels = calculate_continuity(r, units, vel, mass)
print("pos shape: ", pos.shape)
print("vel shape: ", vel.shape)
print("r shape:", r.shape)
print("units shape:", units.shape)
print("conts shape:", conts.shape)
print("dkernels shape:", dkernels.shape)

print("Creating dataset...")
hf = h5py.File("log/dataset", mode="w")
hf.create_dataset('posdiff', data=pos)
hf.create_dataset('norms', data=r)
hf.create_dataset('units', data=units)
hf.create_dataset('veldiff', data=vel)
hf.create_dataset('continuity', data=conts)
hf.create_dataset('dkernels', data=dkernels)
hf.create_dataset('shape', data=pos.shape)
hf.close()
print("Dataset created as HDF5 file.")
