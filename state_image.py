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

    im = np.zeros((height, width, 4))
    for n, pos in enumerate(pos_list):
        x, z = pos
        vx, vz = vel_list[n]
        i = np.argmin(abs(xr - x))
        j = np.argmin(abs(zr - z))
        im[j, i] = x, z, vx, vz
    return im


def save_image(images, skip=0):
    check_dir("cnn_files/ims/{}".format(images[-1].shape[0]))
    images = images[::skip]
    for i, image in enumerate(images):
        f = plt.figure(figsize=(20, 14))
        plt.imshow(image, cmap="GnBu", origin="lower")
        plt.axis('off')
        plt.savefig("cnn_files/ims/{}/{}.png".format(image.shape[0], i), bbox_inches='tight')
        plt.close(f)


def run(plot=False, *args, **kwags):
    labels_path = "./cnn_files/labels/"
    features_path = "./cnn_files/features/"
    data_path = "./cnn_files/cnn_data/"
    hdf_name = "datafile"
    lb_data = os.listdir(labels_path)
    ft_data = os.listdir(features_path)
    if hdf_name in os.listdir(data_path):
        df = pd.read_csv(data_path + files[0])
        N, n_values = df.shape
        print("Found HDF5 raw data file.")
        hf = h5py.File(data_path + hdf_name, mode="r")
        data = np.array(hf.get('simulation'))
        df = pd.DataFrame(data, columns=["x", "z", "xvel", "zvel", "density", "xacc", "zacc", 'drho'])
        hf.close()
    else:
        files = sorted(files, key=lambda x: float(x.split("c")[1]))
        print("Found {} files.".format(n_files))
        print("Reading {} files...".format(files[0]))
        file = pd.read_csv(data_path + files[0]).values
        N, n_values = file.shape
        data = np.zeros((n_files*N, n_values))
        data[:N] = file
        for i, path in enumerate(files[1:], 1):
            if not i % 100:
                print(i, "/", n_files)
            data[i*N:(i+1)*N] = pd.read_csv(data_path + path).values
        print("   Done!")
        hf = h5py.File(data_path+hdf_name, mode="w")
        hf.create_dataset('simulation', data=data)
        hf.close()
        print("Dataset created as HDF5 file.")
        df = pd.DataFrame(data, columns=["x", "z", "xvel", "zvel", "density", "xacc", "zacc", 'drho'])

    pos = df[["x", "z"]].values.reshape(n_files, N, 2)
    vel = df[["xvel", "zvel"]].values.reshape(n_files, N, 2)
    labels = df["drho"].values

    print("pos.shape: ", pos.shape)
    print("vel.shape: ", vel.shape)

    dom = [[0., 2], [0., 2]]

    cnn_data_dir = "./cnn_files/cnn_data"
    filename = "simulation"
    hf = h5py.File("{}/{}".format(cnn_data_dir, filename), 'w')
    size = (78, 78)
    ims = np.zeros((n_files, size[0], size[1], 4))
    print("Generating data for CNN.")
    for i, vals in enumerate(zip(pos, vel)):
        if not i % 100:
            print(i, '/', n_files)
        p, v = vals
        im = generate_cnn_data(dom, p, v, size)
        ims[i] = im
    if plot:
        print("Generating images...")
        save_image(ims, *args, **kwags)
    hf.create_dataset('features', data=ims)
    hf.create_dataset('labels', data=labels)
    hf.create_dataset('number_of_particles', data=N)
    print("Created", hf)
    hf.close()
    return
