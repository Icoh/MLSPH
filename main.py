import numpy as np
from equations import eos_tait, nnps, calculate_continuity, calculate_accel, calculate_density
from tools import check_dir, plot, check_reflect
from functools import partial
from time import time


check_dir("sim")
# Parameters
dim = 2
ndim = np.array([20,40])
dom = [[0., 2], [0., 2]]
c = 30
eos = partial(eos_tait, c)

# Initialize particle positions (staggered cubic lattice)
px = np.linspace(0.0, 0.5, ndim[0])
pz = np.linspace(0.0, 1.0, ndim[1])
xsp = (px[-1] - px[0]) / ndim[0]
zsp = (pz[-1] - pz[0]) / ndim[1]
size = (350*xsp)**2

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
mass = 0.44 * np.ones(N, dtype=np.float64)
density = 1000 * np.ones(N, dtype=np.float64)
pressure = eos(density)

xpos_half = xpos
zpos_half = zpos
xvel_half = xvel
zvel_half = zvel
density_half = density
pressure_half = pressure

# Run simulation
support = 3
h = zsp * 0.8
dt = 0.00005
tlim = 1
calc_acc = partial(calculate_accel, h, N)
calc_cont = partial(calculate_continuity, h, N)
print("Simulating SPH with {} particles.".format(N))
print("Using  h = {:.5f};  dt = {};  c = {}".format(h, dt, c))
time_range = np.arange(dt, tlim, dt)
tl = time_range.size
start = time()


# Perform first half-step to use leap-frog scheme subsequently. The old values will serve as the previous
# half-step, while the new values will serve as initial setup.
nnp = nnps(support, h, xpos, zpos)
xacc, zacc = calc_acc(xpos, zpos, xvel, zvel, mass, density, pressure, nnp)
drho = calc_cont(xpos, zpos, xvel, zvel, mass, nnp)
xpos = xpos + xvel*dt*0.5
zpos = zpos + zvel*dt*0.5
xvel = xvel + xacc*dt*0.5
zvel = zvel + zacc*dt*0.5
density = density + drho*dt*0.5
pressure = eos(density)

nnp = nnps(support, h, xpos, zpos)
sumden = calculate_density(h, xpos, zpos, mass, nnp)
print("Max neighbours = {}".format(max(map(len, nnp))))
print("Max density from summation = {:.3f}".format(max(sumden)))
plot(xpos, zpos, density, dom, 0, dt, s=size)

for c, t in enumerate(time_range, 1):
    if not c % 100:
        elapsed = time()-start
        plot(xpos, zpos, density, dom, c, dt, s=size)
        print("> Progress = {:.2f}%".format(t/tlim))
        print("  - Max density =", max(density))
        print("  - Max neighbours = {}".format(max(map(len, nnp))))
        print("  - Time elapsed = {:.2f}s".format(elapsed))
        print("  - Estimated time left = {:.2f}s".format((tl-c)*elapsed/c))

    nnp = nnps(support, h, xpos, zpos)
    # Leapfrog scheme: first integrate from previous halfstep, then use this in integrate once again.
    xacc, zacc = calc_acc(xpos_half, zpos_half, xvel_half, zvel_half, mass, density_half, pressure_half, nnp)
    xpos_half = xpos + xvel*dt*0.5
    zpos_half = zpos + zvel*dt*0.5
    xvel_half = xvel + xacc*dt*0.5
    zvel_half = zvel + zacc*dt*0.5
    drho = calc_cont(xpos, zpos, xvel, zvel, mass, nnp)
    density_half = density + drho*dt*0.5
    pressure_half = eos(density_half)

    xacc, zacc = calc_acc(xpos_half, zpos_half, xvel_half, zvel_half, mass, density_half, pressure_half, nnp)
    xvel = xvel + xacc*dt
    zvel = zvel + zacc*dt
    xpos = xpos_half + xvel*dt*0.5
    zpos = zpos_half + zvel*dt*0.5
    drho = calc_cont(xpos, zpos, xvel, zvel, mass, nnp)
    density = density_half + drho*dt*0.5
    pressure = eos(density)

    xpos, xvel = check_reflect(xpos, xvel, dom[0])
    zpos, zvel = check_reflect(zpos, zvel, dom[1])
