import numpy as np
from equations import eos_tait, nnps, calculate_continuity, calculate_accel, calculate_density
from tools import check_dir, save_data, plot, wall_gen
from functools import partial
from time import time
import csv


check_dir("sim")
check_dir("log")
# Parameters
dim = 2
dom = [[0., 2], [0., 2]]
C = 30
eos = partial(eos_tait, C)

# Initialize particle positions (staggered cubic lattice)
ndim = np.array([14, 28])
px = np.linspace(0.75, 1.25, ndim[0])
pz = np.linspace(0.0, 1., ndim[1])
xsp = (px[-1] - px[0]) / ndim[0]
zsp = (pz[-1] - pz[0]) / ndim[1]
size = (600 * xsp)**1.5

xpos, zpos = np.meshgrid(px, pz)
xpos, zpos = xpos.ravel(), zpos.ravel()
N = xpos.size

# Generate wall of particles:
w = 2
w1 = wall_gen([dom[0][0]-w*xsp, dom[0][1]+w*xsp], [-zsp*w, -zsp], xsp, zsp)
w4 = wall_gen([dom[0][0]-w*xsp, dom[0][1]+w*xsp], [dom[1][1]+zsp, dom[1][1]+w*zsp], xsp, zsp)
w2 = wall_gen([-w*xsp, -xsp], dom[1], xsp, zsp)
w3 = wall_gen([dom[0][1]+xsp, dom[0][1]+xsp*w], dom[1], xsp, zsp)
xwall = np.concatenate((w1[0], w2[0], w3[0], w4[0]), axis=0)
zwall = np.concatenate((w1[1], w2[1], w3[1], w4[1]), axis=0)

real_particles = np.array([True for _ in xpos] + [False for _ in xwall])
xpos = np.concatenate((xpos, xwall), axis=0)
zpos = np.concatenate((zpos, zwall), axis=0)

# State
N_all = xpos.size
xvel = np.zeros(N_all, dtype=np.float64)
zvel = np.zeros(N_all, dtype=np.float64)
mass = 1.37 * np.ones(N_all, dtype=np.float64)
density = 1000 * np.ones(N_all, dtype=np.float64)
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
tlim = 0.5

print("Simulating SPH with {} particles.".format(N))
print("Using  h = {:.5f};  dt = {};  c = {}".format(h, dt, C))
time_range = np.arange(dt, tlim, dt)
tl = time_range.size
calc_acc = partial(calculate_accel, h, N_all)
calc_cont = partial(calculate_continuity, h, N_all)
start = time()


# Perform first half-step to use leap-frog scheme subsequently. The old values will serve as the previous
# half-step, while the new values will serve as initial setup.
nnp = nnps(support, h, xpos, zpos)
xacc, zacc = calc_acc(xpos, zpos, xvel, zvel, mass, density, pressure, nnp[real_particles])
drho, _, _, _ = calc_cont(xpos, zpos, xvel, zvel, mass, nnp)
xpos = xpos + xvel * dt * 0.5
zpos = zpos + zvel * dt * 0.5
xvel = xvel + xacc * dt * 0.5
zvel = zvel + zacc * dt * 0.5
density = density + drho * dt * 0.5
pressure = eos(density)

nnp = nnps(support, h, xpos, zpos)
sumden = calculate_density(h, xpos, zpos, mass, nnp)[real_particles]
# density[real_particles] = sumden
print("Neighbours count range: {} - {}".format(min(map(len, nnp)), max(map(len, nnp))))
print("Density range from summation: {:.3f} - {:.3f}".format(min(sumden), max(sumden)))
plot(xpos, zpos, density, dom, 0, dt, s=size)
# save_data(0, xpos, zpos, xvel, zvel, density, xacc[real_particles],
#           zacc[real_particles], drho)

for c, t in enumerate(time_range, 1):
    nnp = nnps(support, h, xpos, zpos)
    # Leapfrog scheme: first integrate from previous halfstep, then use this in integrate once again.
    xacc, zacc= calc_acc(xpos_half, zpos_half, xvel_half, zvel_half,
                          mass, density_half, pressure_half, nnp[real_particles])
    xpos_half = xpos + xvel * dt * 0.5
    zpos_half = zpos + zvel * dt * 0.5
    xvel_half = xvel + xacc * dt * 0.5
    zvel_half = zvel + zacc * dt * 0.5
    drho, _, _, _ = calc_cont(xpos, zpos, xvel, zvel, mass, nnp)
    density_half = density + drho * dt * 0.5
    pressure_half = eos(density_half)

    xacc, zacc = calc_acc(xpos_half, zpos_half, xvel_half, zvel_half,
                          mass, density_half, pressure_half, nnp[real_particles])
    xvel = xvel + xacc * dt
    zvel = zvel + zacc * dt
    xpos = xpos_half + xvel * dt * 0.5
    zpos = zpos_half + zvel * dt * 0.5
    drho, dists, xvdiffs, zvdiffs = calc_cont(xpos, zpos, xvel, zvel, mass, nnp)
    density = density_half + drho * dt * 0.5
    pressure = eos(density)

    if not c % 100:
        elapsed = time() - start
        plot(xpos, zpos, density, dom, c, dt, s=size)
        nnsize = list(map(len, nnp))
        print("> Progress = {:.2f}%".format(t / tlim * 100))
        print("  - Density range: {:.3f} - {:.3f}".format(min(density), max(density)))
        print("  - Neighbours count range: {} - {}".format(min(nnsize), max(nnsize)))
        print("  - Time elapsed: {:.2f}s".format(elapsed))
        print("  - ETA: {:.2f}s".format((tl - c) * elapsed / c))

    with open("log/posdiff/t{}.csv".format(c), "w+") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(dists)
    with open("log/xvdiff/t{}.csv".format(c), "w+") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(xvdiffs)
    with open("log/zvdiff/t{}.csv".format(c), "w+") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(zvdiffs)
    with open("log/drho/t{}.csv".format(c), "w+") as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerows(drho.reshape((-1, 1)))
