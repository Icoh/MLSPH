import matplotlib.pyplot as plt
import numpy as np
import os
import errno


'''
General misc tools
'''


def check_dir(path):
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def plot(X, Z, C, domain, step, dt, *args, **kwargs):
    f = plt.figure(figsize=(20, 14))
    plt.scatter(X, Z, c=C, cmap='summer', alpha=0.6, *args, **kwargs)
    plt.clim(0, 0.15)
    plt.colorbar()
    plt.title("T = {:.3f} s".format(step*dt))
    # plt.xlim(domain[1][0]-0.05, domain[1][-1]+0.05)
    # plt.ylim(domain[1][0]-0.05, domain[1][-1]+0.05)
    f.axes[0].axis('equal')
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
    xrange = np.arange(xdom[0], xdom[1]+xspacing, xspacing)
    zrange = np.arange(zdom[0], zdom[1]+zspacing, zspacing)
    X, Z = np.meshgrid(xrange, zrange)
    wall_x, wall_z = list(), list()
    for x, z in zip(X.ravel(), Z.ravel()):
        wall_x.append(x)
        wall_z.append(z)
    return np.array(wall_x), np.array(wall_z)
