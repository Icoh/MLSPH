import numpy as np


def gaussian(r, unit_vect, h, dim=2):
    q = r / h
    g = np.exp(-q ** 2) / (h ** 2 * np.pi) ** (dim / 2.)
    dg = np.zeros((1,2))
    if isinstance(unit_vect, np.ndarray):
        gv, qv = g[:], q[:]
        gv.shape = (gv.size, 1)
        qv.shape = (qv.size, 1)
        r.shape = (r.size, 1)
        dg = -2 * qv / h * gv * unit_vect
    return g, dg