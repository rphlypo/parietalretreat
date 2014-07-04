import numpy as np
from scipy import linalg


def random_diagonal(s, d_min=0., d_max=1.):
    """Generates random diagonal matrix, with diagonal elements in range
    [d_min, d_max].

    Parameters
    ==========
    s: int
        The first dimension of the array.
    d_min: float, optional
        Lower bound for the diagonal elements.
    d_max: float, optional
        Upper bound for the diagonal elements.

    Returns
    =======
    diag: array
        2D output diaogonal array, shape (s0, s0).
    """
    diag = np.random.rand(s) * (d_max - d_min) + d_min
    return np.diag(diag)


def random_diagonal_spd(s, d_min=1., d_max=2.):
    """Generates random positive definite diagonal matrix"""
    assert(d_min > 0)
    assert(d_max > 0)
    return random_diagonal(s, d_min, d_max)


def random_spd(s, eig_min=1.0, eig_max=2.0):
    """Generates random symmetric positive definite matrix with eigenvalues in
    a given range."""
    ran = np.random.rand(s, s)
    q, _ = linalg.qr(ran)
    d = random_diagonal_spd(s, eig_min, eig_max)
    return q.dot(d).dot(q.T)


def random_non_singular(s):
    """Generates random non singular matrix"""
    d = random_diagonal_spd(s)
    ran1 = np.random.rand(s, s)
    ran2 = np.random.rand(s, s)
    u, _ = linalg.qr(ran1)
    v, _ = linalg.qr(ran2)
    return u.dot(d).dot(v.T)