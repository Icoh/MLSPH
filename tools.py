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


def save_data(c, xpos, zpos, xvel, zvel, density, xacc, zacc):
    with open("log/c{}.csv".format(c), "w+") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["x", "z", "xvel", "zvel", "density", "xacc", "zacc"])
        writer.writerows(zip(xpos, zpos, xvel, zvel, density, xacc, zacc))


def plot(X, Z, D, domain, step, dt, *args, **kwargs):
    f = plt.figure(figsize=(20, 14))
    plt.scatter(X, Z, c=D, cmap='viridis_r', alpha=0.6, *args, **kwargs)
    plt.clim(950, 1050)
    plt.colorbar()
    plt.title("T = {:.3f} s".format(step*dt))
    plt.xlim(domain[0][0] - 0.1, domain[0][-1] + 0.1)
    plt.ylim(domain[1][0] - 0.1, domain[1][-1] + 0.1)
    plt.savefig("{}/{}.png".format("sim", str(step)), bbox_inches='tight')
    plt.close(f)
    return


def unit(vector, norm):
    vector = np.array(vector)

    nm = norm[:]
    nm.shape = (nm.size, 1)

    unit_vect = vector / nm
    nans = np.isnan(unit_vect)
    unit_vect[nans] = [0, 0]
    return unit_vect


def damp_reflect(pos, vel, wall):
    damp = 0.75

    quiet = vel == 0.0
    p_pos = pos

    tb = (pos - wall) / vel
    pos -= vel * (1 - damp) * tb

    pos = 2 * wall - pos
    vel = -vel
    vel *= damp

    vel[quiet] = np.zeros(vel[quiet].size)
    pos[quiet] = p_pos[quiet]
    return pos, vel


def check_reflect(pos, vel, domain):
    lower = domain[0]
    upper = domain[1]
    islower = pos < lower
    isupper = pos > upper
    if any(islower):
        pos[islower], vel[islower] = damp_reflect(pos[islower], vel[islower], lower)
    if any(isupper):
        pos[isupper], vel[isupper] = damp_reflect(pos[isupper], vel[isupper], upper)
    return pos, vel
