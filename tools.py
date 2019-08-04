import matplotlib.pyplot as plt
import numpy as np
import os
import csv


def check_dir(name):
    current = os.listdir()
    if name in current:
        return
    print("{} folder not found. Created new one.".format(name))
    os.mkdir(name)
    return


def save_data(c, xpos, zpos, xvel, zvel, density, xacc, zacc, drho):
    with open("log/features/f{}.csv".format(c), "w+") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["x", "z", "xvel", "zvel"])
        writer.writerows(zip(xpos, zpos, xvel, zvel))
    with open("log/labels/l{}.csv".format(c), "w+") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["density", "xacc", "zacc", 'drho'])
        writer.writerows(zip(density, xacc, zacc, drho))


def save_dist(dist, veldiff, c):
    with open("log/nnp/t{}.csv".format(c), "w+") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["r", "v"])
        writer.writerows(zip(dist, veldiff))


def save_drho(drho, c):
    with open("log/drho/t{}.csv".format(c), "w+") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["drho"])
        writer.writerows(drho)


def plot(X, Z, C, domain, step, dt, *args, **kwargs):
    f = plt.figure(figsize=(20, 14))
    plt.scatter(X, Z, c=C, cmap='viridis_r', alpha=0.6, *args, **kwargs)
    plt.clim(940, 1060)
    plt.colorbar()
    plt.title("T = {:.3f} s".format(step*dt))
    plt.xlim(domain[0][0] - 0.1, domain[0][-1] + 0.1)
    plt.ylim(domain[1][0] - 0.1, domain[1][-1] + 0.1)
    plt.savefig("{}/{}.png".format("sim", step), bbox_inches='tight')
    plt.close(f)
    return


def unit(vector, norm):
    vector = np.array(vector)

    nm = norm[:]
    nm.shape = (nm.size, 1)

    unit_vect = vector / nm
    # nans = np.isnan(unit_vect)
    # unit_vect[nans] = np.zeros(1)
    return unit_vect


def wall_gen(xdom, zdom, xspacing, zspacing):
    xrange = np.arange(xdom[0], xdom[1]+0.9*xspacing, xspacing)
    zrange = np.arange(zdom[0], zdom[1]+0.9*zspacing, zspacing)
    X, Z = np.meshgrid(xrange, zrange)
    wall_x, wall_z = list(), list()
    for x, z in zip(X.ravel(), Z.ravel()):
        wall_x.append(x)
        wall_z.append(z)
    return np.array(wall_x), np.array(wall_z)
