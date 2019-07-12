import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lin
import scipy.spatial as sp


# Tools
def plot(X, Z, D, domain, step, dt):
    f = plt.figure(figsize=(20, 14))
    plt.scatter(X, Z, c=D, cmap='viridis_r', alpha=0.6)
    plt.clim(950, 1050)
    plt.colorbar()
    plt.title("T = {:.4f} s".format(step*dt))
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
        return pos, vel
    elif any(isupper):
        pos[isupper], vel[isupper] = damp_reflect(pos[isupper], vel[isupper], upper)
        return pos, vel
    else:
        return pos, vel


# Kernel
def gaussian(r, unit_vect, h):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    gv, qv = g[:], q[:]
    gv.shape = (gv.size, 1)
    qv.shape = (qv.size, 1)
    dg = -2 * qv / h * gv * unit_vect
    return g, dg


# Equations
def eos(rho):
    c = 10.
    gamma = 7.
    rho0 = 1000.
    b = c ** 2 * rho0 / gamma
    p = b * ((rho / rho0) ** gamma - 1)
    return p


def summation_density(mass, kernel):
    return mass * kernel


def continuity(mass, vdiff, dkernel):
    return mass * np.sum(vdiff * dkernel, axis=1)


def pressure_term(mass, rhoa, pressa, rhob, pressb, dkernel):
    a = pressa / rhoa ** 2
    b = pressb / rhob ** 2
    c = (a + b)
    c.shape = (c.size, 1)
    mass.shape = (mass.size, 1)
    result = -mass * c * dkernel
    return result


# Parameters
dim = 2
ndim = np.array([21, 21])

# Initialize
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

