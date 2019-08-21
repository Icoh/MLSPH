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


def define_poiseuille(k, h, nu):
    def poiseuille(z):
        return k*(h*z - z**2)/(2*nu)
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

# poise = define_poiseuille(0.25, 0.4, 0.014)
# z = np.linspace(0, 0.4, 100)
# v = poise(z)
#
# plt.plot(v, z)
# plt.show()

nb = find_max_nb(["log/xdiff/t22870.csv"])
diffs_df = pd.read_csv("log/xdiff/t9820.csv", names=[i for i in range(nb)]).fillna(0)
print(diffs_df.head(9))
diffs = diffs_df.values
for diff in diffs:
    plt.hist(diff, bins=20)
    plt.show()
