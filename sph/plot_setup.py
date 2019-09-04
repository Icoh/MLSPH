import numpy as np
from sph.equations import eos_tait, nnps
from sph.tools import wall_gen
from functools import partial
import matplotlib.pyplot as plt


'''
Code to plot initial particle setup, with different colors
for each type of particle in the simulation.
'''


# Parameters
dim = 2
dom = [[0., 0.1], [0., 0.4]]
C = 30
eos = partial(eos_tait, C)
nx, nz = 10, 40
m = 0.11365
rho0 = 1000

# Initialize particle positions
px = np.linspace(0.0, 0.1, nx)
pz = np.linspace(0.0, 0.4, nz)
xsp = (px[-1] - px[0]) / (nx-1)
zsp = (pz[-1] - pz[0]) / (nz-1)
scatter_size = (2000 * xsp) ** 1.5
xpos, zpos = np.meshgrid(px, pz)
xpos, zpos = xpos.ravel(), zpos.ravel()
N_real = xpos.size

# Generate wall of particles:
w = 2
w1 = wall_gen([dom[0][0], dom[0][1]], [-w*zsp, -zsp], xsp, zsp)
w4 = wall_gen([dom[0][0], dom[0][1]], [dom[1][1] + zsp, dom[1][1] + w * zsp], xsp, zsp)
xwall = np.concatenate((w1[0], w4[0]), axis=0)
zwall = np.concatenate((w1[1], w4[1]), axis=0)
N_wall = xwall.size

xtot = np.concatenate((xpos, xwall), axis=0)
ztot = np.concatenate((zpos, zwall), axis=0)

# Periodic arrays
def periodize(x, z):
    xper = np.concatenate((x, x + dom[0][1] + xsp, x - dom[0][1] - xsp), axis=0)
    zper = np.concatenate((z, z, z), axis=0)
    return xper, zper


# Run simulation
h = zsp * 0.9
support = 3

xp, zp = periodize(xpos, zpos)
xwp, zwp = periodize(xwall, zwall)
xtp, ztp = periodize(xtot, ztot)
nnp = nnps(support, h, xtp, ztp)

i = 0
nbs = nnp[i]
i_x, i_z = xtp[i], ztp[i]
j_x, j_z = xtp[nbs], ztp[nbs]

k = 159
nbs = nnp[k]
k_x, k_z = xtp[k], ztp[k]
l_x, l_z = xtp[nbs], ztp[nbs]

f = plt.figure(figsize=(20, 14))
plt.scatter(xwp, zwp, c='black', label='Pared')
plt.scatter(xp, zp, c='rosybrown', label='Copias')
plt.scatter(xpos, zpos, c='firebrick', label='Reales')
plt.scatter(j_x, j_z, c='green', label='Vecinos')
plt.scatter(i_x,i_z, c='orange', label='Objetivos')
plt.scatter(l_x, l_z, c='green')
plt.scatter(k_x,k_z, c='orange')
f.axes[0].axis('equal')
plt.legend()
plt.show()
