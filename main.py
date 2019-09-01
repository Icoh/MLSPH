import numpy as np
from equations import eos_tait, nnps, calculate_continuity, calculate_accel, calculate_density
from tools import check_dir, save_data, plot, wall_gen
from functools import partial
from time import time
import matplotlib.pyplot as plt
import csv
import os, errno


def define_poiseuille(k, H, nu):
    def poiseuille(z):
        return k*(H*z - z**2)/(2*nu)
    return poiseuille


check_dir("sim")
check_dir("log")
paths = ["log/params", "log/inputs", "log/targets", "log/poise"]
for path in paths:
    try:
        os.mkdir(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

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
mass = m * np.ones(N_all, dtype=np.float64)
density = rho0 * np.ones(N_all, dtype=np.float64)
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
h = zsp * 0.9
support = 3
dt = 0.0001
tlim = 30
with open("log/params/values.csv".format(0), "w+") as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["h", "support", "dt", "tlim", "C", "mass"])
    writer.writerow([h, support, dt, tlim, C, m])

print("Simulating SPH with {} particles.".format(N_real))
print("Using  h = {:.5f};  dt = {};  c = {}".format(h, dt, C))
time_range = np.arange(dt, tlim, dt)
tl = time_range.size
calc_acc = partial(calculate_accel, h, N_all)
calc_cont = partial(calculate_continuity, h, N_all)
start = time()

xp, zp, xvp, zvp, mp, dp, pp = periodize(xpos, zpos, xvel, zvel, mass, density, pressure)
nnp = nnps(support, h, xp, zp)
sumden = calculate_density(h, xp, zp, mp, nnp)[:N_real]
density[:N_real] = sumden
dp[:N_real] = sumden
print("Neighbours count range: {} - {}".format(min(map(len, nnp[:N_real])), max(map(len, nnp))))
print("Density range from summation: {:.3f} - {:.3f}".format(min(sumden), max(sumden)))

with open("log/inputs/t{}.csv".format(0), "w+") as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["xpos", "zpos", "xvel", "zvel"])
    writer.writerows(zip(xp, zp, xvp, zvp))
with open("log/targets/t{}.csv".format(0), "w+") as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["xacc", "zacc", "density", "drho"])
    zeros = np.zeros_like(density)
    writer.writerows(zip(zeros, zeros, density, zeros))
plot(xp, zp, np.sqrt(xvp**2 + zvp**2), dom, 0, dt, s=scatter_size)

# Perform first half-step to use leap-frog scheme subsequently. The old values will serve as the previous
# half-step, while the new values will serve as initial setup.
N_sim = N_real + N_wall
nnp = nnps(support, h, xp, zp)
xacc, zacc = calc_acc(xp, zp, xvp, zvp, mp, dp, pp, nnp[:N_real])
drho = calc_cont(xp, zp, xvp, zvp, mp, nnp[:N_sim])
xpos = xpos + xvel * dt * 0.5
zpos = zpos + zvel * dt * 0.5
xvel = xvel + xacc * dt * 0.5
zvel = zvel + zacc * dt * 0.5
density = density + drho * dt * 0.5
pressure = eos(density)
xp, zp, xvp, zvp, mp, dp, pp = periodize(xpos, zpos, xvel, zvel, mass, density, pressure)

c = 0
try:
    for c, t in enumerate(time_range, 1):
        # Save initial positions and velocities (right before performing first time step)
        if not c % 50:
            with open("log/inputs/t{}.csv".format(c), "w+") as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["xpos", "zpos", "xvel", "zvel"])
                writer.writerows(zip(xp, zp, xvp, zvp))

        # Leap-frog scheme: first integrate from previous halfstep, then use this to integrate once again.
        # First leap-frog step
        drho = calc_cont(xp, zp, xvp, zvp, mp, nnp[:N_sim])
        xacc, zacc = calc_acc(*periodize(xpos_half, zpos_half, xvel_half, zvel_half, mass, density_half, pressure_half),
                              nnp[:N_real])
        xpos_half = xpos + xvel * dt * 0.5
        zpos_half = zpos + zvel * dt * 0.5
        xvel_half = xvel + xacc * dt * 0.5
        zvel_half = zvel + zacc * dt * 0.5
        density_half = density + drho * dt * 0.5
        pressure_half = eos(density_half)

        # Second leap-frog step
        xacc, zacc = calc_acc(*periodize(xpos_half, zpos_half, xvel_half, zvel_half, mass, density_half, pressure_half),
                              nnp[:N_real])
        xvel = xvel + xacc * dt
        zvel = zvel + zacc * dt
        xpos = xpos_half + xvel * dt * 0.5
        zpos = zpos_half + zvel * dt * 0.5

        # When particles leave the right-side domain, they are restored to the left-side domain.
        out = xpos > dom[0][1]+xsp
        xpos[out] = xpos[out]-(dom[0][1]+xsp)

        xp, zp, xvp, zvp, mp, _, _ = periodize(xpos, zpos, xvel, zvel, mass, density, pressure)
        nnp = nnps(support, h, xp, zp)
        drho = calc_cont(xp, zp, xvp, zvp, mp, nnp[:N_sim])
        density = density_half + drho * dt * 0.5
        pressure = eos(density)

        # Save rate of changes at the end of timestep corresponding to the initial positions and velocities
        if not c % 50:
            with open("log/poise/t{}.csv".format(c), "w+") as file:
                writer = csv.writer(file)
                writer.writerows(zip(xvel, zpos))
            with open("log/targets/t{}.csv".format(c), "w+") as file:
                writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["xacc", "zacc", "density", "drho"])
                writer.writerows(zip(xacc, zacc, density, drho))

        if not c % 100:
            elapsed = time() - start
            plot(xpos, zpos, np.sqrt(xvel**2 + zvel**2), dom, c, dt, s=scatter_size)
            nnsize = list(map(len, nnp[:N_real]))
            print("> Progress = {:.2f}%".format(t / tlim * 100))
            print("  - Density range: {:.3f} - {:.3f}".format(min(density), max(density)))
            print("  - Neighbours count range: {} - {}".format(min(nnsize), max(nnsize)))
            print("  - Max x velocity: {:.3f}".format(max(xvel)))
            print("  - Time elapsed: {:.2f}s".format(elapsed))
            print("  - ETA: {:.2f}s".format((tl - c) * elapsed / c))
except KeyboardInterrupt:
    print("Early manually interrupted.")

with open("log/poise/t{}.csv".format(c), "w+") as file:
    writer = csv.writer(file)
    writer.writerows(zip(xvel, zpos))


plt.plot(xvel, zpos, 'k.')

poise = define_poiseuille(k=0.05, H=0.4, nu=C*h/(8*2.5))
z = np.linspace(0, 0.4, 100)
v = poise(z)
plt.plot(v, z)
plt.show()
