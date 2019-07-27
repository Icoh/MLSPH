import numpy as np
import pandas as pd
from sklearn import preprocessing
from tools import check_dir
import matplotlib.pyplot as plt
import os
import h5py


def pos_to_index(domain, im_array, pos_list, value_list):
    xdom = domain[0]
    ydom = domain[1]
    height, width = im_array.shape
    xr = np.linspace(xdom[0], xdom[1], width)
    yr = np.linspace(ydom[0], ydom[1], height)

    for n, pos in enumerate(pos_list):
        x, y = pos
        i = np.argmin(abs(xr-x))
        j = np.argmin(abs(yr-y))
        im_array[j, i] = value_list[n]
    return im_array


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

    pos = df[["x", "z"]].values
    pos = np.array(list(zip(pos[:, 0], pos[:, 1]))).reshape(n_files, N, 2)
    vel = df[["xvel", "zvel"]].values
    vel = ((vel[:, 0]**2 + vel[:, 1]**2)**0.5).reshape(n_files, N)
    vel = preprocessing.normalize(vel)
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
        hf.create_dataset('shape', data=[n_files, N, ])
        hf.create_dataset('labels', data=labels)
        sim_data = []
        for i, vals in enumerate(zip(pos, vel)):
            p, v = vals
            im = np.zeros((100, 100))
            im = pos_to_index(dom, im, p, v)
            sim_data.append(im)
            # f = plt.figure(figsize=(20, 14))
            # plt.imshow(im, cmap="GnBu", origin="lower")
            # plt.axis('off')
            # plt.savefig("cnn_data_ims/{}/{}.png".format(im.shape[0], i), bbox_inches='tight')
            # plt.close(f)
        sim_data = np.array(sim_data)
        hf.create_dataset('features', data=sim_data)
        print("Created", hf)
        hf.close()
    else:
        print(filename, "file already exists.")
    return
