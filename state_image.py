import numpy as np
import pandas as pd
from sklearn import preprocessing
from tools import check_dir
import matplotlib.pyplot as plt
import os
import h5py
import re


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
    lb_files = os.listdir(labels_path)
    ft_files = os.listdir(features_path)
    ft_files = sorted(ft_files, key=lambda x: float(re.findall('\d+', x)[0]))
    lb_files = sorted(lb_files, key=lambda x: float(re.findall('\d+', x)[0]))
    n_files = len(lb_files)
    if hdf_name in os.listdir(data_path):
        df_labels = pd.read_csv(labels_path + lb_files[0])
        df_features = pd.read_csv(features_path + ft_files[0])
        N_lab, lb_values = df_labels.shape
        N_feat, ft_values = df_features.shape
        print("Found HDF5 raw data file.")
        hf = h5py.File(data_path + hdf_name, mode="r")
        feats = np.array(hf.get('features'))
        labs = np.array(hf.get('labels'))
        df_labels = pd.DataFrame(labs, columns=["density", "xacc", "zacc", 'drho'])
        df_features = pd.DataFrame(feats, columns=["x", "z", "xvel", "zvel"])
        hf.close()
    else:
        print("Found {} files.".format(n_files))
        print("Reading {} files...".format(n_files*2))
        lab0 = pd.read_csv(labels_path + lb_files[0])
        feat0 = pd.read_csv(features_path + ft_files[0])
        N_lab, lb_values = lab0.shape
        N_feat, ft_values = feat0.shape
        labs = np.zeros((n_files*N_lab, lb_values))
        feats = np.zeros((n_files*N_feat, ft_values))
        labs[:N_lab], feats[:N_feat] = lab0, feat0
        for i, path in enumerate(ft_files[1:], 1):
            if not i % 100:
                print(i, "/", n_files)
            feats[i*N_feat:(i+1)*N_feat] = pd.read_csv(features_path + path).values
        for i, path in enumerate(lb_files[1:], 1):
            if not i % 100:
                print(i, "/", n_files)
            labs[i*N_lab:(i+1)*N_lab] = pd.read_csv(labels_path + path).values
        print("   Done!")
        hf = h5py.File(data_path+hdf_name, mode="w")
        hf.create_dataset('features', data=feats)
        hf.create_dataset('labels', data=labs)
        hf.close()
        print("Dataset created as HDF5 file.")
        df_labels = pd.DataFrame(labs, columns=["density", "xacc", "zacc", 'drho'])
        df_features = pd.DataFrame(feats, columns=["x", "z", "xvel", "zvel"])

    pos = df_features[["x", "z"]].values.reshape(n_files, N_feat, 2)
    vel = df_features[["xvel", "zvel"]].values.reshape(n_files, N_feat, 2)
    drho = df_labels["drho"].values

    print("pos.shape: ", pos.shape)
    print("vel.shape: ", vel.shape)

    dom = [[0., 2], [0., 2]]

    cnn_data_dir = "./cnn_files/cnn_data"
    filename = "training"
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
    hf.create_dataset('labels', data=drho)
    hf.create_dataset('number_of_particles', data=N_lab)
    print("Created", hf)
    hf.close()
    return
