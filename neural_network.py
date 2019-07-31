import numpy as np
from equations import eos_tait, nnps, calculate_continuity, calculate_accel, calculate_density
from tools import check_dir, save_data, plot, wall_gen
from functools import partial
from time import time
from tensorflow.contrib.keras import models
import state_image as sti


check_dir("sim")
check_dir("log")
# Parameters
im_size = (-1, 78, 78, 4)
dim = 2
dom = [[0., 2], [0., 2]]
C = 50
eos = partial(eos_tait, C)

# Initialize particle positions (staggered cubic lattice)
ndim = np.array([15, 30])
px = np.linspace(0.0, 0.5, ndim[0])
pz = np.linspace(0.0, 1., ndim[1])
xsp = (px[-1] - px[0]) / ndim[0]
zsp = (pz[-1] - pz[0]) / ndim[1]
size = (600 * xsp)**1.5

xpos, zpos = np.meshgrid(px, pz)
xpos, zpos = xpos.ravel(), zpos.ravel()
N = xpos.size

# Generate wall of particles:
w1 = wall_gen([dom[0][0]-2*xsp, dom[0][1]+2*xsp], [-zsp * 2.1, -zsp], xsp, zsp)
w4 = wall_gen([dom[0][0]-2*xsp, dom[0][1]+2*xsp], [dom[1][1], dom[1][1]+1.9*zsp], xsp, zsp)
w2 = wall_gen([-xsp, -2.1 * xsp], dom[1], -xsp, zsp)
w3 = wall_gen([dom[0][1] + xsp, dom[0][1] + xsp * 2], dom[1], xsp, zsp)
xwall = np.concatenate((w1[0], w2[0], w3[0], w4[0]), axis=0)
zwall = np.concatenate((w1[1], w2[1], w3[1], w4[1]), axis=0)

real_particles = np.array([True for _ in xpos] + [False for _ in xwall])
xpos = np.concatenate((xpos, xwall), axis=0)
zpos = np.concatenate((zpos, zwall), axis=0)

# State
N_all = xpos.size
xvel = np.zeros(N_all, dtype=np.float64)
zvel = np.zeros(N_all, dtype=np.float64)
mass = 0.77 * np.ones(N_all, dtype=np.float64)
density = 1000 * np.ones(N_all, dtype=np.float64)
pressure = eos(density)

xpos_half = xpos
zpos_half = zpos
xvel_half = xvel
zvel_half = zvel
density_half = density
pressure_half = pressure

# Run simulation
model = models.load_model('model.h5')
support = 3
h = zsp * 0.8
dt = 0.00005
tlim = 1.
calc_acc = partial(calculate_accel, h, N_all)
calc_cont = partial(calculate_continuity, h, N_all)
print("Simulating SPH with {} particles.".format(N))
print("Using  h = {:.5f};  dt = {};  c = {}".format(h, dt, C))
time_range = np.arange(dt, tlim, dt)
tl = time_range.size
start = time()

# Perform first half-step to use leap-frog scheme subsequently. The old values will serve as the previous
# half-step, while the new values will serve as initial setup.
nnp = nnps(support, h, xpos, zpos)[real_particles]
xacc, zacc = calc_acc(xpos, zpos, xvel, zvel, mass, density, pressure, nnp)
drho = calc_cont(xpos, zpos, xvel, zvel, mass, nnp)
xpos = xpos + xvel * dt * 0.5
zpos = zpos + zvel * dt * 0.5
xvel = xvel + xacc * dt * 0.5
zvel = zvel + zacc * dt * 0.5
density = density + drho * dt * 0.5
pressure = eos(density)

nnp = nnps(support, h, xpos, zpos)[real_particles]
sumden = calculate_density(h, xpos, zpos, mass, nnp)[real_particles]
# density[real_particles] = sumden
print("Neighbours count range: {} - {}".format(min(map(len, nnp)), max(map(len, nnp))))
print("Density range from summation: {:.3f} - {:.3f}".format(min(sumden), max(sumden)))
plot(xpos, zpos, density, dom, 0, dt, s=size)
# save_data(0, xpos, zpos, xvel, zvel, density, xacc, zacc)

for c, t in enumerate(time_range, 1):
    if not c % 10:
        elapsed = time() - start
        plot(xpos, zpos, density, dom, c, dt, s=size)
        nnsize = list(map(len, nnp))
        print("> Progress = {:.2f}%".format(t / tlim * 100))
        print("  - Density range: {:.3f} - {:.3f}".format(min(density), max(density)))
        print("  - Neighbours count range: {} - {}".format(min(nnsize), max(nnsize)))
        print("  - Time elapsed = {:.2f}s".format(elapsed))
        print("  - ETA = {:.2f}s".format((tl - c) * elapsed / c))
    # if not c % 10:
    #     save_data(c, xpos, zpos, xvel, zvel, density, xacc, zacc)

    nnp = nnps(support, h, xpos, zpos)[real_particles]
    # Leapfrog scheme: first integrate from previous halfstep, then use this in integrate once again.
    xacc, zacc = calc_acc(xpos_half, zpos_half, xvel_half, zvel_half, mass, density_half, pressure_half, nnp)
    xpos_half = xpos + xvel * dt * 0.5
    zpos_half = zpos + zvel * dt * 0.5
    xvel_half = xvel + xacc * dt * 0.5
    zvel_half = zvel + zacc * dt * 0.5
    state = sti.generate_cnn_data(dom, zip(xpos_half, zpos_half),
                                  list(zip(xvel_half, zvel_half)), size=im_size[1:3]).reshape(*im_size)
    drho[real_particles] = model.predict(state)[0][real_particles]
    density_half = density + drho * dt * 0.5
    pressure_half = eos(density_half)

    xacc, zacc = calc_acc(xpos_half, zpos_half, xvel_half, zvel_half, mass, density_half, pressure_half, nnp)
    xvel = xvel + xacc * dt
    zvel = zvel + zacc * dt
    xpos = xpos_half + xvel * dt * 0.5
    zpos = zpos_half + zvel * dt * 0.5
    state = sti.generate_cnn_data(dom, zip(xpos, zpos),
                                  list(zip(xvel, zvel)), size=im_size[1:3]).reshape(*im_size)
    drho = model.predict(state)[0]
    density = density_half + drho * dt * 0.5
    pressure = eos(density)
