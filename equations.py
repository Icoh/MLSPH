import numpy as np
import scipy.linalg as lin
import scipy.spatial as sp
from kernel import gaussian
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
    return mass * np.sum(vdiff * dkernel, axis=1)


def pressure_term(mass, rhoa, pressa, rhob, pressb, dkernel):
    a = pressa / rhoa ** 2
    b = pressb / rhob ** 2
    c = (a + b)
    c.shape = (c.size, 1)
    mass.shape = (mass.size, 1)
    return -mass * c * dkernel


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

        posdiff = np.array(list(zip(i_x - j_x, i_z - j_z)))
        r = lin.norm(posdiff, axis=1)
        posunit = unit(posdiff, r)
        veldiff = np.array(list(zip(i_xv - j_xv, i_zv - j_zv)))

        kn, dkn = gaussian(r, posunit, h)
        acc = sum(pressure_term(j_mass, i_rho, i_pressure, j_rho, j_pressure, dkn)) + np.array([0, -10])
        xa[i] = acc[0]
        za[i] = acc[1]
    return xa, za


def calculate_continuity(h, N, x0, z0, xv0, zv0, m0, nn_list):
    ddens = np.zeros(N, dtype=np.float64)
    xdist = list()
    zdist = list()
    xvdiff = list()
    zvdiff = list()

    for i, nbs in enumerate(nn_list):
        i_x, i_z = x0[i], z0[i]
        i_xv, i_zv = xv0[i], zv0[i]

        j_x, j_z = x0[nbs], z0[nbs]
        j_xv, j_zv = xv0[nbs], zv0[nbs]
        j_mass = m0[nbs]

        posdiff = np.array(list(zip(i_x - j_x, i_z - j_z)))
        r = lin.norm(posdiff, axis=1)
        posunit = unit(posdiff, r)
        veldiff = np.array(list(zip(i_xv - j_xv, i_zv - j_zv)))

        xdist.append(posdiff[0])
        zdist.append(posdiff[1])
        xvdiff.append(veldiff[:, 0])
        zvdiff.append(veldiff[:, 1])

        kn, dkn = gaussian(r, posunit, h)
        ddens[i] = sum(continuity(j_mass, veldiff, dkn))
    return ddens, xdist, zdist, xvdiff, zvdiff


def calculate_density(h, x, z, mass, nn_list):
    dens = np.zeros(mass.size)
    for i, nbs in enumerate(nn_list):
        i_x, i_z = x[i], z[i]
        i_mass = mass[i]

        j_x, j_z = x[nbs], z[nbs]
        j_mass = mass[nbs]

        posdiff = np.array(list(zip(i_x - j_x, i_z - j_z)))
        r = lin.norm(posdiff, axis=1)
        posunit = unit(posdiff, r)

        kn, dkn = gaussian(r, posunit, h)
        kn0, _ = gaussian(0, 0, h)
        dens[i] = sum(summation_density(j_mass, kn)) + kn0*i_mass

        # plt.scatter(x0, z0, c='red')
        # plt.scatter(j_x, j_z, c='green')
        # plt.scatter(i_x,i_z, c='blue')
        # plt.show()
    return dens
