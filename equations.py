import numpy as np
import scipy.linalg as lin
import scipy.spatial as sp
from kernel import gaussian
from tools import unit


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
    return -mass * c * dkernel


def calculate(support, h, N, x0, z0, xv0, zv0, m0, dens0, press0):
    """
    Performs first step to be able to use leap-frog integration subsequently. Thus, this step will
    consist on integrating just a half-step in time, so we can use the initial conditions for the next step.
    """
    # Initialize data arrays
    pos = list(zip(x0, z0))
    xa = np.zeros(N, dtype=np.float64)
    za = np.zeros(N, dtype=np.float64)
    ddens = np.zeros(N, dtype=np.float64)

    # Find nearest neighbours and calculate new density and acceleration
    kdt = sp.cKDTree(pos)
    max_dist = support * h
    nnp_all = kdt.query_ball_point(kdt.data, max_dist)
    for i, nbs in enumerate(nnp_all):
        i_x, i_z = x0[i], z0[i]
        i_xv, i_zv = xv0[i], zv0[i]
        i_mass = m0[i]
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
        ddens[i] = sum(continuity(j_mass, veldiff, dkn))
        acc = sum(pressure_term(j_mass, i_rho, i_pressure, j_rho, j_pressure, dkn)) + np.array([0, -10])
        xa[i] = acc[0]
        za[i] = acc[1]
    return xa, za, ddens
