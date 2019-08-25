import numpy as np
import scipy.linalg as lin
import scipy.spatial as sp
from kernel import kernel
from tools import unit


def eos_tait(c, rho):
    gamma = 7.
    rho0 = 1000.
    b = c ** 2 * rho0 / gamma
    p = b * ((rho / rho0) ** gamma - 1)
    return p


def summation_density(mass, kernel):
    return mass * kernel


def continuity(mass, vdiff, dkernel):
    return mass * np.sum(vdiff * dkernel, axis=-1)


def pressure_term(mass, rhoa, pressa, rhob, pressb, dkernel):
    a = pressa / rhoa ** 2
    b = pressb / rhob ** 2
    c = (a + b)
    c.shape = (c.size, 1)
    mass.shape = (mass.size, 1)
    return -mass * c * dkernel


def artif_visc(h, mass, dist, r, vdiff, rhoa, rhob, dkernel):
    alpha = 1.
    beta = 0
    c = 30.
    dot = np.sum(dist*vdiff, axis=-1)
    trues = dot < 0
    mu = (h * dot/(r.ravel()**2 + 0.01 * h**2))[trues]
    drho = (rhoa + rhob)[trues] / 2
    visc = np.zeros_like(dot)
    visc[trues] = ((-alpha * c * mu) + beta * mu ** 2)/drho
    visc.shape = (visc.size, 1)
    return -mass * visc * dkernel


def nnps(support, h, xpos, zpos):
    pos = list(zip(xpos, zpos))
    kdt = sp.cKDTree(pos)
    max_dist = support * h
    nnp_all = kdt.query_ball_point(kdt.data, max_dist)
    for i, nbs in enumerate(nnp_all):
        nbs.remove(i)
    return nnp_all


def calculate_accel(h, N, x0, z0, xv0, zv0, m0, dens0, press0, nn_list):
    xa = np.zeros(N, dtype=np.float64)
    za = np.zeros(N, dtype=np.float64)

    for i, nbs in enumerate(nn_list):
        i_x, i_z = x0[i], z0[i]
        i_xv, i_zv = xv0[i], zv0[i]
        i_rho = dens0[i]
        i_pressure = press0[i]

        j_x, j_z = x0[nbs], z0[nbs]
        j_xv, j_zv = xv0[nbs], zv0[nbs]
        j_mass = m0[nbs]
        j_rho = dens0[nbs]
        j_pressure = press0[nbs]

        posdiff = np.dstack((i_x - j_x, i_z - j_z)).reshape(-1,2)
        r = lin.norm(posdiff, axis=-1)
        posunit = unit(posdiff, r)
        veldiff = np.dstack((i_xv - j_xv, i_zv - j_zv)).reshape(-1,2)

        kn, dkn = kernel(r, posunit, h)
        acc = sum(pressure_term(j_mass, i_rho, i_pressure, j_rho, j_pressure, dkn)) + np.array([+0.05, 0])
        acc += sum(artif_visc(h, j_mass, posdiff, r, veldiff, i_rho, j_rho, dkn))
        xa[i] = acc[0]
        za[i] = acc[1]
    return xa, za


def calculate_continuity(h, N, x0, z0, xv0, zv0, m0, nn_list):
    ddens = np.zeros(N, dtype=np.float64)

    for i, nbs in enumerate(nn_list):
        i_x, i_z = x0[i], z0[i]
        i_xv, i_zv = xv0[i], zv0[i]

        j_x, j_z = x0[nbs], z0[nbs]
        j_xv, j_zv = xv0[nbs], zv0[nbs]
        j_mass = m0[nbs]

        posdiff = np.dstack((i_x - j_x, i_z - j_z)).reshape(-1,2)
        r = lin.norm(posdiff, axis=-1)
        posunit = unit(posdiff, r)
        veldiff = np.dstack((i_xv - j_xv, i_zv - j_zv)).reshape(-1,2)

        _, dkn = kernel(r, posunit, h)
        ddens[i] = sum(continuity(j_mass, veldiff, dkn))
    return ddens


def calculate_density(h, x0, z0, mass, nn_list):
    dens = np.zeros(mass.size)
    for i, nbs in enumerate(nn_list):
        i_x, i_z = x0[i], z0[i]
        i_mass = mass[i]

        j_x, j_z = x0[nbs], z0[nbs]
        j_mass = mass[nbs]

        posdiff = np.dstack((i_x - j_x, i_z - j_z)).reshape(-1,2)
        r = lin.norm(posdiff, axis=-1)
        posunit = unit(posdiff, r)

        kn, dkn = kernel(r, posunit, h)
        kn0, _ = kernel(0, 0, h)
        dens[i] = sum(summation_density(j_mass, kn)) + kn0*i_mass

        # Plot nearest neighbors
        # plt.scatter(x0, z0, c='red')
        # plt.scatter(j_x, j_z, c='green')
        # plt.scatter(i_x,i_z, c='blue')
        # plt.show()
    return dens
