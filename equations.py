import numpy as np


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