import numpy as np
import scipy.linalg as lin
import scipy.spatial as sp
from equations import eos, continuity, pressure_term
from kernel import gaussian
from tools import unit, plot, check_reflect


# Parameters
dim = 2
ndim = np.array([21, 21])

# Initialize particle positions (staggered cubic lattice)
dom = [[0., 2.], [0., 2.]]
px = np.linspace(0.0, 1.0, ndim[0])
pz = np.linspace(0.0, 1.0, ndim[1])
xsp = (px[-1] - px[0]) / ndim[0]
zsp = (pz[-1] - pz[0]) / ndim[1]

xpos, zpos = [], []
for nbs, z in enumerate(pz):
    offset = nbs * ndim[0]
    if not nbs % 2:
        for i, x in enumerate(px):
            xpos.append(x), zpos.append(z)
    else:
        for i, x in enumerate(px[:-1]):
            xpos.append(x + xsp / 2), zpos.append(z)

# State
xpos = np.array(xpos)
zpos = np.array(zpos)
N = xpos.size

xvel = np.zeros(N, dtype=np.float64)
zvel = np.zeros(N, dtype=np.float64)
xacc = np.zeros(N, dtype=np.float64)
zacc = np.zeros(N, dtype=np.float64)
drho = np.zeros(N, dtype=np.float64)

mass = 2.5 * np.ones(N, dtype=np.float64)
density = 1000 * np.ones(N, dtype=np.float64)
pressure = eos(density)

# Run simulation
support = 2
h = zsp * 1.45
dt = 0.0001
tlim = 1
print("Simulating SPH with {} particles.".format(N))
print("Using h: {}".format(h))
pos = list(zip(xpos, zpos))

for c, t in enumerate(np.arange(0, tlim, dt)):
    if not c % 50:
        plot(xpos, zpos, density, dom, c, dt)
        print("Max density=", max(density))

    # Find nearest neighbours and calculate new density and acceleration
    kdt = sp.cKDTree(pos)
    max_dist = support * h
    nnp_all = kdt.query_ball_point(kdt.data, max_dist)
    for i, nbs in enumerate(nnp_all):
        i_x, i_z = xpos[i], zpos[i]
        i_xv, i_zv = xvel[i], zvel[i]
        i_mass = mass[i]
        i_rho = density[i]
        i_pressure = pressure[i]

        j_x, j_z = xpos[nbs], zpos[nbs]
        j_xv, j_zv = xvel[nbs], zvel[nbs]
        j_mass = mass[nbs]
        j_rho = density[nbs]
        j_pressure = pressure[nbs]

        posdiff = np.array(list(zip(i_x - j_x, i_z - j_z)))
        r = lin.norm(posdiff, axis=1)
        posunit = unit(posdiff, r)
        veldiff = np.array(list(zip(i_xv - j_xv, i_zv - j_zv)))

        kn, dkn = gaussian(r, posunit, h)
        # density[i] = sum(summation_density(j_mass, kn))
        drho[i] = sum(continuity(j_mass, veldiff, dkn))
        acc = sum(pressure_term(j_mass, i_rho, i_pressure, j_rho, j_pressure, dkn)) + np.array([0,-10])
        xacc[i] = acc[0]
        zacc[i] = acc[1]

        # plt.scatter(xpos, zpos, c='red')
        # plt.scatter(j_x, j_z, c='green')
        # plt.scatter(i_x, i_z, c='blue')
        # plt.show()

    # Integrate
    density += drho * dt
    pressure = eos(density)
    xvel += xacc * dt
    zvel += zacc * dt
    xpos += xvel * dt
    zpos += zvel * dt

    xpos, xvel = check_reflect(xpos, xvel, dom[0])
    zpos, zvel = check_reflect(zpos, zvel, dom[1])

