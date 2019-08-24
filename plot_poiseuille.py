import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import csv


# log_path = "log/state"
#
# state_paths = sorted(os.listdir(log_path), key=lambda x: float(re.findall('\d+', x)[0]))
#
# state_df = pd.read_csv(log_path + state_paths[-1])
# xvel = state_df["xvel"].values
# zpos = state_df["zpos"].values


def define_poiseuille(k, H, nu):
    def poiseuille(z):
        return k*(H*z - z**2)/(2*nu)
    return poiseuille


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


c = 30
alpha = 1
support = 3
h = 0.01103
nu = alpha*c*h/8
H = 0.4
k = 0.05/0.19526

poise = define_poiseuille(k, H, nu)
z = np.linspace(0, H, 100)
v = poise(z)

files = sorted(os.listdir("log/poise"), key=lambda x: float(re.findall('\d+', x)[0]))
nb = find_max_nb(["log/poise/{}".format(files[0])])

for i, file in enumerate(files):
    print(i)
    df = pd.read_csv("log/poise/{}".format(file), names=[i for i in range(nb)]).fillna(0)
    xpos = df[0].values
    zvel = df[1].values

    fig = plt.figure()
    plt.title(file)
    plt.plot(v, z)
    plt.plot(xpos, zvel, 'k.')
    plt.savefig("log/poise_plots/{}".format(i), bbox_inches='tight')
    plt.close(fig)
