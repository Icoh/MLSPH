import numpy as np
from equations import eos, calculate
from tools import plot, check_reflect
from functools import partial
from time import time


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

mass = 2.5 * np.ones(N, dtype=np.float64)
density = 1000 * np.ones(N, dtype=np.float64)
pressure = eos(density)

xvel = np.zeros(N, dtype=np.float64)
zvel = np.zeros(N, dtype=np.float64)
xacc_pre = np.zeros(N, dtype=np.float64)
zacc_pre = np.zeros(N, dtype=np.float64)
rho_pre = density

# Run simulation
support = 2
h = zsp * 1.45
dt = 0.0001
tlim = 1
evo = partial(calculate, support, h, N)
print("Simulating SPH with {} particles.".format(N))
print("Using  h = {:.5f};  dt = {}".format(h, dt))

time_range = np.arange(0, tlim, dt)
tl = time_range.size
start = time()
for c, t in enumerate(time_range):
    if not c % 100:
        elapsed = time()-start
        plot(xpos, zpos, density, dom, c, dt)
        print("> Progress = {:.2f}%".format(t/tlim))
        print("  - Max density =", max(density))
        if c:
            print("  - Time elapsed = {:.2f}s".format(elapsed))
            print("  - Estimated time left = {:.2f}s".format((tl-c)*elapsed/c))

    xacc, zacc, drho = evo(xpos, zpos, xvel, zvel, mass, density, pressure)

    # Integrate
    density += drho * dt
    pressure = eos(density)
    xvel += xacc * dt
    zvel += zacc * dt
    xpos += xvel * dt
    zpos += zvel * dt

    xpos, xvel = check_reflect(xpos, xvel, dom[0])
    zpos, zvel = check_reflect(zpos, zvel, dom[1])

