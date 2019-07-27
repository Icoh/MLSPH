import numpy as np
import pandas as pd
from sklearn import preprocessing
from tools import check_dir
import matplotlib.pyplot as plt
import os
import h5py


def generate_cnn_data(domain, pos_list, vel_list, size=(64, 64)):
    xdom = domain[0]
    ydom = domain[1]
    height, width = size
    xr = np.linspace(xdom[0], xdom[1], width)
    zr = np.linspace(ydom[0], ydom[1], height)

    im = np.zeros((height, width, 2))
    for n, pos in enumerate(pos_list):
        x, z = pos
        vx, vz = vel_list[n]
        i = np.argmin(abs(xr-x))
        j = np.argmin(abs(zr-z))
        im[j, i] = vx, vz
    return im


def run():
    data_dir = "./data/"
    files = os.listdir(data_dir)
    n_files = len(files)
    hdf_name = "datafile"
    if hdf_name in files:
        n_files -= 1
        df = pd.read_csv(data_dir + files[0])
        N, n_values = df.shape
        print("Found HDF5 file.")
        df = pd.read_hdf(data_dir + hdf_name, 'simulation')
    else:
        files = sorted(files, key=lambda x: float(x.split("c")[1]))
        print("Found {} files.".format(n_files))
        print("Reading files...".format(files[0]), end='')
        df = pd.read_csv(data_dir + files[0])
        N, n_values = df.shape
        for path in files[1:]:
            dft = pd.read_csv(data_dir + path)
            df = df.append(dft)
        print("   Done!")
        df.to_hdf("./data/datafile", key='simulation', mode="w")

    pos = df[["x", "z"]].values.reshape(n_files, N, 2)
    vel = df[["xvel", "zvel"]].values.reshape(n_files, N, 2)
    labels = df["density"].values

    print("pos.shape: ", pos.shape)
    print("vel.shape: ", vel.shape)

    dom = [[0., 2], [0., 2]]

    cnn_data_dir = "cnn_data"
    filename = "simulation"
    check_dir(cnn_data_dir)
    files = os.listdir(cnn_data_dir)
    if filename not in files:
        hf = h5py.File("{}/{}".format(cnn_data_dir, filename), 'w')
        ims = list()
        for i, vals in enumerate(zip(pos, vel)):
            p, v = vals
            im = generate_cnn_data(dom, p, v)
            ims.append(im)
            # f = plt.figure(figsize=(20, 14))
            # plt.imshow(im, cmap="GnBu", origin="lower")
            # plt.axis('off')
            # plt.savefig("cnn_data_ims/{}/{}.png".format(im.shape[0], i), bbox_inches='tight')
            # plt.close(f)
        hf.create_dataset('velocity', data=ims)
        hf.create_dataset('density', data=labels)
        hf.create_dataset('number_of_particles', data=N)
        print("Created", hf)
        hf.close()
    else:
        print(filename, "file already exists.")
    return
