import numpy as np
from equations import eos_tait, nnps, calculate_accel, calculate_density, kernel
from tools import check_dir, unit, plot, wall_gen
from functools import partial
from time import time
from tensorflow.contrib.keras import models
import state_image as sti
import scipy.linalg as lin
import os, errno
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt


def calculate_continuity(h, N, x0, z0, xv0, zv0, m0, nn_list):
    ddens = np.zeros(N)
    X = np.zeros((len(nn_list), 20, 5))

    for i, nbs in enumerate(nn_list):

        i_x, i_z = x0[i], z0[i]
        i_xv, i_zv = xv0[i], zv0[i]

        j_x, j_z = x0[nbs], z0[nbs]
        j_xv, j_zv = xv0[nbs], zv0[nbs]

        posdiff = np.array(list(zip(i_x - j_x, i_z - j_z)))
        veldiff = np.array(list(zip(i_xv - j_xv, i_zv - j_zv)))
        r = lin.norm(posdiff, axis=-1).reshape(-1, 1)
        units = posdiff/r

        X[i, :len(nbs), 0] = r.ravel()
        X[i, :len(nbs), 1] = units[:, 0]
        X[i, :len(nbs), 2] = units[:, 1]
        X[i, :len(nbs), 3] = veldiff[:, 0]
        X[i, :len(nbs), 4] = veldiff[:, 1]
    cont = model.predict(X)
    ddens = np.sum(cont, axis=-2)
    return np.array(ddens).ravel()


def define_poiseuille(k, h, nu):
    def poiseuille(z):
        return k*(h*z - z**2)/(2*nu)
    return poiseuille


check_dir("sim")
check_dir("log")
try:
    logs = ["drho", "xdiff", "zdiff", "xvdiff", "zvdiff"]
    for path in logs:
        os.mkdir("log/" + path)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

# Parameters
dim = 2
dom = [[0., 0.2], [0., 0.4]]
C = 30
eos = partial(eos_tait, C)

# Initialize particle positions (staggered cubic lattice)
ndim = np.array([16, 32])
px = np.linspace(0.0, 0.2, ndim[0])
pz = np.linspace(0.0, 0.4, ndim[1])
xsp = (px[-1] - px[0]) / ndim[0]
zsp = (pz[-1] - pz[0]) / ndim[1]
size = (1000 * xsp) ** 1.5

xpos, zpos = np.meshgrid(px, pz)
xpos, zpos = xpos.ravel(), zpos.ravel()
N_real = xpos.size

# Generate wall of particles:
w = 2
w1 = wall_gen([dom[0][0], dom[0][1]], [-zsp * w, -zsp], xsp, zsp)
w4 = wall_gen([dom[0][0], dom[0][1]], [dom[1][1] + zsp, dom[1][1] + w * zsp], xsp, zsp)
# w2 = wall_gen([-w * xsp, -xsp], dom[1], xsp, zsp)
# w3 = wall_gen([dom[0][1] + xsp, dom[0][1] + xsp * w], dom[1], xsp, zsp)
xwall = np.concatenate((w1[0], w4[0]), axis=0)
zwall = np.concatenate((w1[1], w4[1]), axis=0)
N_wall = xwall.size

xpos = np.concatenate((xpos, xwall), axis=0)
zpos = np.concatenate((zpos, zwall), axis=0)

# State
N_all = xpos.size
xvel = np.zeros(N_all, dtype=np.float64)
zvel = np.zeros(N_all, dtype=np.float64)
mass = 0.1497 * np.ones(N_all, dtype=np.float64)
density = 1000 * np.ones(N_all, dtype=np.float64)
pressure = eos(density)

xpos_half = xpos
zpos_half = zpos
xvel_half = xvel
zvel_half = zvel
density_half = density
pressure_half = pressure


# Periodic arrays
def periodize(x, z, xv, zv, m, d, p):
    xper = np.concatenate((x, x + dom[0][1] + xsp, x - dom[0][1] - xsp), axis=0)
    zper = np.concatenate((z, z, z), axis=0)
    xvper = np.concatenate((xv, xv, xv), axis=0)
    zvper = np.concatenate((zv, zv, zv), axis=0)
    mper = np.concatenate((m, m, m), axis=0)
    dper = np.concatenate((d, d, d), axis=0)
    pper = np.concatenate((p, p, p), axis=0)
    return xper, zper, xvper, zvper, mper, dper, pper


# Run simulation
model = models.load_model('nn_cont.h5')
support = 4
h = zsp * 0.6
dt = 0.00005
tlim = 1.5

print("Simulating SPH with {} particles.".format(N_real))
print("Using  h = {:.5f};  dt = {};  c = {}".format(h, dt, C))
time_range = np.arange(dt, tlim, dt)
tl = time_range.size
calc_acc = partial(calculate_accel, h, N_all)
calc_cont = partial(calculate_continuity, h, N_all)
start = time()

# Perform first half-step to use leap-frog scheme subsequently. The old values will serve as the previous
# half-step, while the new values will serve as initial setup.
xp, zp, xvp, zvp, mp, dp, pp = periodize(xpos, zpos, xvel, zvel, mass, density, pressure)
# plot(xp, zp, dp, dom, 0, dt, s=size)
real_particles = np.array([True for _ in range(N_real)] + [False for _ in range(xp.size - N_real)])
sim_particles = np.array([True for _ in range(N_real + N_wall)] + [False for _ in range(xp.size - N_real - N_wall)])
nnp = nnps(support, h, xp, zp)
xacc, zacc = calc_acc(xp, zp, xvp, zvp, mp, dp, pp, nnp[real_particles])
drho = calc_cont(xp, zp, xvp, zvp, mp, nnp[sim_particles])
xpos = xpos + xvel * dt * 0.5
zpos = zpos + zvel * dt * 0.5
xvel = xvel + xacc * dt * 0.5
zvel = zvel + zacc * dt * 0.5
density = density + drho * dt * 0.5
pressure = eos(density)

nnp = nnps(support, h, xpos, zpos)
sumden = calculate_density(h, xpos, zpos, mass, nnp)[:N_real]
# density[real_particles] = sumden
print("Neighbours count range: {} - {}".format(min(map(len, nnp[:N_real])), max(map(len, nnp))))
print("Density range from summation: {:.3f} - {:.3f}".format(min(sumden), max(sumden)))
plot(xp, zp, dp, dom, 0, dt, s=size)
max_nn_count = 0
max_vel = 0

try:
    for c, t in enumerate(time_range, 1):
        # Leapfrog scheme: first integrate from previous halfstep, then use this in integrate once again.
        nnp = nnps(support, h, xp, zp)
        drho= calc_cont(xp, zp, xvp, zvp, mp, nnp[sim_particles])
        density_half = density + drho * dt * 0.5
        xacc, zacc = calc_acc(*periodize(xpos_half, zpos_half, xvel_half, zvel_half, mass, density_half, pressure_half),
                              nnp[real_particles])
        xpos_half = xpos + xvel * dt * 0.5
        zpos_half = zpos + zvel * dt * 0.5
        xvel_half = xvel + xacc * dt * 0.5
        zvel_half = zvel + zacc * dt * 0.5
        pressure_half = eos(density_half)

        xacc, zacc = calc_acc(*periodize(xpos_half, zpos_half, xvel_half, zvel_half, mass, density_half, pressure_half),
                              nnp[real_particles])
        xvel = xvel + xacc * dt
        zvel = zvel + zacc * dt
        xpos = xpos_half + xvel * dt * 0.5
        zpos = zpos_half + zvel * dt * 0.5
        xp, zp, xvp, zvp, mp, _, _ = periodize(xpos, zpos, xvel, zvel, mass, density, pressure)
        drho = calc_cont(xp, zp, xvp, zvp, mp, nnp[sim_particles])
        density = density_half + drho * dt * 0.5
        pressure = eos(density)

        out = xpos > dom[0][1]+xsp
        xpos[out] = xpos[out]-(dom[0][1]+xsp)

        if not c % 100:
            elapsed = time() - start
            plot(xpos, zpos, density, dom, c, dt, s=size)
            nnsize = list(map(len, nnp[:N_real]))
            print("> Progress = {:.2f}%".format(t / tlim * 100))
            print("  - Density range: {:.3f} - {:.3f}".format(min(density), max(density)))
            print("  - Neighbours count range: {} - {}".format(min(nnsize), max(nnsize)))
            print("  - Time elapsed: {:.2f}s".format(elapsed))
            print("  - ETA: {:.2f}s".format((tl - c) * elapsed / c))

except KeyboardInterrupt:
    plt.plot(xvel, zpos, 'k.')

    poise = define_poiseuille(0.25, 0.4, 0.014)
    z = np.linspace(0, 0.4, 100)
    v = poise(z)
    plt.plot(v, z)
    plt.show()

plt.plot(xvel, zpos, 'k.')

poise = define_poiseuille(0.25, 0.4, 0.014)
z = np.linspace(0, 0.4, 100)
v = poise(z)
plt.plot(v, z)
plt.show()
