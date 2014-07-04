import numpy as np
from scipy import linalg


def random_diagonal(shape, d_min=0., d_max=1.):
    """Generates random diagonal matrix, with elements in the range
    [d_min, d_max]
    """
    d = np.random.rand(shape) * (d_max - d_min) + d_min
    return np.diag(d)


def random_diagonal_spd(shape, d_min=1., d_max=2.):
    """Generates random positive definite diagonal matrix"""
    assert(d_min > 0)
    assert(d_max > 0)
    return random_diagonal(shape, d_min, d_max)


def random_spd(shape, eig_min=1.0, eig_max=2.0):
    """Generates random symmetric positive definite matrix"""
    ran = np.random.rand(shape, shape)
    q, _ = linalg.qr(ran)
    d = random_diagonal_spd(shape, eig_min, eig_max)
    return q.dot(d).dot(q.T)


def random_non_singular(shape):
    """Generates random non singular matrix"""
    d = random_diagonal_spd(shape)
    ran1 = np.random.rand(shape, shape)
    ran2 = np.random.rand(shape, shape)
    u, _ = linalg.qr(ran1)
    v, _ = linalg.qr(ran2)
    return u.dot(d).dot(v.T)


def random_wishart(dof, V, size=None):
    """Draw random matrix from Wishart distribution"""
    assert(dof > V.shape[0] - 1)
    assert(isspd(V))
    if size is None:
        X = np.random.multivariate_normal(0, V, dof)
        S = (X.T).dot(X)
    else:
        S = np.empty((size,) + V.shape)
        for k in xrange(size):
            S[k] = random_wishart(dof, V, None)
    return S

def random_generalized_normal(mean, variance):
    """Draw random matrix from generalized Gaussian distribution"""