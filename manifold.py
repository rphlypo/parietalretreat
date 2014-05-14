import warnings

import numpy as np
from numpy.testing import assert_array_less, assert_array_almost_equal
from scipy import linalg


def my_stack(arrays):
    return np.concatenate([a[np.newaxis] for a in arrays])


def sqrtm(mat):
    """ Matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs = linalg.eigh(mat)
    return np.dot(vecs * np.sqrt(vals), vecs.T)


def inv_sqrtm(mat):
    """ Inverse of matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs = linalg.eigh(mat)
    return np.dot(vecs / np.sqrt(vals), vecs.T)


def inv(mat):
    """ Inverse of matrix, for symmetric positive definite matrices.
    """
    vals, vecs = linalg.eigh(mat)
    return np.dot(vecs / vals, vecs.T)


def logm(mat):
    """ Logarithm of matrix, for symetric positive definite matrices
    """
    vals, vecs = linalg.eigh(mat)
    return np.dot(vecs * np.log(vals), vecs.T)


def expm(mat):
    """ Exponential of matrix, for real symetric matrices
    """
    try:
        assert_array_almost_equal(mat, mat.T)
        assert(np.all(np.isreal(mat)))
    except AssertionError:
        raise ValueError("at least one matrix is not real symmetric")
    vals, vecs = linalg.eigh(mat)
    return np.dot(vecs * np.exp(vals), vecs.T)


def tangent_space_norm(v, p):
    """ Norm of vector v in the tangent space at point p"""
    p_inv = inv(p)
    return np.sqrt(np.trace(p_inv.dot(v).dot(p_inv).dot(v)))


def log_map(x, displacement, mean=False):
    """ The Riemannian log map at point 'displacement'.

    See Algorithm 2 of:
        P. Thomas Fletcher, Sarang Joshi. Riemannian Geometry for the
        Statistical Analysis of Diffusion Tensor Data. Signal Processing, 2007.
    """
    vals, vecs = linalg.eigh(displacement)
    sqrt_vals = np.sqrt(vals)
    whitening = (vecs / sqrt_vals).T
    vals_y, vecs_y = linalg.eigh(whitening.dot(x).dot(whitening.T))
    sqrt_displacement = (vecs * sqrt_vals).dot(vecs_y)
    return (sqrt_displacement * np.log(vals_y)).dot(sqrt_displacement.T)


def frechet_mean(mats, max_iter=10, tol=1e-3, adaptative=False):
    """ Computes Frechet mean of a list of symmetric positive definite
    matrices.

    Minimization of the objective function by an intrinsic gradient descent in
    the manifold: moving from the current point fre to the next one is
    done along a short geodesic arc in the opposite direction of the covariant
    derivative of the objective function evaluated at point fre.

    See Algorithm 3 of:
        P. Thomas Fletcher, Sarang Joshi. Riemannian Geometry for the
        Statistical Analysis of Diffusion Tensor Data. Signal Processing, 2007.

    Parameters
    ==========
    mats: list of array
        list of symmetric positive definite matrices, same shape.
    max_iter: int, optional
        maximal number of iterations.
    tol: float, optional
        tolerance.

    Returns
    =======
    fre: array
        Frechet mean of the matrices.
    """
    # Real, symmetry and positive definiteness check
    for mat in mats:
        try:
            assert_array_almost_equal(mat, mat.T)
            assert(np.all(np.isreal(mat)))
            assert_array_less(0.0, np.linalg.eigvalsh(mat))
        except AssertionError:
            raise ValueError("at least one matrix is not real spd")

    mats = my_stack(mats)

    # Initialization
    fre = np.mean(mats, axis=0)
    tolerance_reached = False
    norm_old = np.inf
    step = 1.
    for n in xrange(max_iter):
        vals_fre, vecs_fre = linalg.eigh(fre)
        fre_inv_sqrt = (vecs_fre / np.sqrt(vals_fre)).dot(vecs_fre.T)
        eighs = [linalg.eigh(fre_inv_sqrt.dot(mat).dot(fre_inv_sqrt)) for
                 mat in mats]

        # Log map of mats[n] at point fre is
        # sqrtm(fre).dot(logms[n]).dot(sqrtm(fre))
        logms = [(vecs * np.log(vals)).dot(vecs.T) for vals, vecs in eighs]

        # Covariant derivative is fre.dot(logms_mean)
        logms_mean = np.mean(logms, axis=0)
        try:
            assert np.all(np.isfinite(logms_mean))
        except AssertionError:
            raise FloatingPointError("Nan value after logarithm operation")

        vals_log, vecs_log = linalg.eigh(logms_mean)

        # Move along the geodesic with stepsize step
        fre_sqrt = (vecs_fre * np.sqrt(vals_fre)).dot(vecs_fre.T)
        fre = fre_sqrt.dot(
            vecs_log * np.exp(vals_log * step)).dot(vecs_log.T).dot(fre_sqrt)

        # Norm of the covariant derivative on the tangent space at point fre
        norm = np.sqrt(np.trace(logms_mean.dot(logms_mean)))
        if tol is not None and norm < tol:
            tolerance_reached = True
            break

        if norm > norm_old:
            step = step / 2.
            norm = norm_old
        if norm < norm_old and adaptative:
            step = 2. * step

    if tol is not None and not tolerance_reached:
        warnings.warn("Maximum number of iterations reached without")  # +\
#                      " getting to the requested tolerance level.")

    return fre


def grad_frechet_mean(mats, max_iter=10, tol=1e-3, adaptative=True):
    """ Returns at each iteration step of the frechet_mean algorithm the norm
    of the covariant derivative. Norm is intrinsic norm on the tangent space at
    the Frechet mean at the current step.

    Parameters
    ==========
    mats: list of array
        list of symmetric positive definite matrices, same shape.
    max_iter: int, optional
        maximal number of iterations.
    tol: float, optional
        tolerance.

    Returns
    =======
    grad_norm: list of float
        Norm of the covariant derivative in the tangent space at each step.
    """
    # Real, symmetry and positive definiteness check
    for mat in mats:
        print mat.shape
        try:
#            assert(is_spd(mat)) # TODO: replace by assert(is_spd(mat)) and test
            assert_array_almost_equal(mat, mat.T)
            assert(np.all(np.isreal(mat)))
            assert_array_less(0.0, np.linalg.eigvalsh(mat))
        except AssertionError:
            raise ValueError("at least one matrix is not real spd")

    mats = my_stack(mats)

    # Initialization
    fre = np.mean(mats, axis=0)
    norm_old = np.inf
    step = 1.
    tolerance_reached = False
    grad_norm = []
    for n in xrange(max_iter):
        vals_fre, vecs_fre = linalg.eigh(fre)
        fre_inv_sqrt = (vecs_fre / np.sqrt(vals_fre)).dot(vecs_fre.T)
        eighs = [linalg.eigh(fre_inv_sqrt.dot(mat).dot(fre_inv_sqrt)) for
                 mat in mats]

        # Log map of mats[n] at point fre is
        # sqrtm(fre).dot(logms[n]).dot(sqrtm(fre))
        logms = [(vecs * np.log(vals)).dot(vecs.T) for vals, vecs in eighs]

        # Covariant derivative is fre.dot(logms_mean)
        logms_mean = np.mean(logms, axis=0)
        try:
            assert np.all(np.isfinite(logms_mean))
        except AssertionError:
            raise FloatingPointError("Nan value after logarithm operation")

        vals_log, vecs_log = linalg.eigh(logms_mean)

        # Move along the geodesic with stepsize step
        fre_sqrt = (vecs_fre * np.sqrt(vals_fre)).dot(vecs_fre.T)
        fre = fre_sqrt.dot(
            vecs_log * np.exp(vals_log * step)).dot(vecs_log.T).dot(fre_sqrt)

        # Norm of the covariant derivative on the tangent space at point fre
        norm = np.sqrt(np.trace(logms_mean.dot(logms_mean)))
        grad_norm.append(norm)
        if tol is not None and norm < tol:
            tolerance_reached = True
            break

        if norm > norm_old:
            step = step / 2.
            norm = norm_old

    if tol is not None and not tolerance_reached:
        warnings.warn("Maximum number of iterations reached without")  # +\
#                      " getting to the requested tolerance level.")

    return grad_norm


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


def is_spd(M, decimal=15):
    """Assert that input matrix is real symmetric positive definite.

    M must be symmetric down to specified decimal places and with no complex
    entry.
    The check is performed by checking that all eigenvalues are positive.

    Parameters
    ==========
    M: numpy.ndarray
        matrix.

    Returns
    =======
    answer: boolean
        True if matrix is symmetric real positive definite, False otherwise.
    """
    if not np.allclose(M, M.T, atol=0.1 ** decimal):
        print("matrix not symmetric to {0} decimals".format(decimal))
        return False
    if np.all(np.iscomplex(M)):
        print("matrix has a non real value {0}".format(M[np.iscomplex(M)][0]))
    eigvalsh = np.linalg.eigvalsh(M)
    ispd = eigvalsh.min() > 0
    if not ispd:
        print("matrix has a negative eigenvalue: %.3f" % eigvalsh.min())
    return ispd