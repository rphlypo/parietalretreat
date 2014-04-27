import warnings

import numpy as np
from numpy.testing import assert_array_less
from scipy import linalg
from scipy.linalg.lapack import get_lapack_funcs


# Bypass scipy for faster eigh (and dangerous: Nan will kill it)
my_eigh, = get_lapack_funcs(('syevr', ), np.zeros(1))


def my_stack(arrays):
    return np.concatenate([a[np.newaxis] for a in arrays])


def sqrtm(mat):
    """ Matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs * np.sqrt(vals), vecs.T)


def inv_sqrtm(mat):
    """ Inverse of matrix square-root, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs / np.sqrt(vals), vecs.T)


def inv(mat):
    """ Inverse of matrix, for symetric positive definite matrices.
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs / vals, vecs.T)


def logm(mat):
    """ Logarithm of matrix, for symetric positive definite matrices
    """
    vals, vecs, success_flag = my_eigh(mat)
    return np.dot(vecs * np.log(vals), vecs.T)


def expm(mat):
    """ Exponential of matrix, for symetric positive definite matrices
    """
    vals, vecs, success_flag = my_eigh(mat)
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
    vals, vecs, success_flag = my_eigh(displacement)
    sqrt_vals = np.sqrt(vals)
    whitening = (vecs / sqrt_vals).T
    vals_y, vecs_y, success_flag = my_eigh(whitening.dot(x).dot(whitening.T))
    sqrt_displacement = (vecs * sqrt_vals).dot(vecs_y)
    return (sqrt_displacement * np.log(vals_y)).dot(sqrt_displacement.T)


def frechet_mean(mats, max_iter=10, tol=1e-3):
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
    # Positive definiteness check
    for mat in mats:
        assert_array_less(0.0, my_eigh(mat)[0])

    mats = my_stack(mats)

    # Initialization
    fre = np.mean(mats, axis=0)
    tolerance_reached = False
    norm_old = np.inf
    step = 1.
    for n in xrange(max_iter):
        vals_fre, vecs_fre, success_flag = my_eigh(fre)
        fre_inv_sqrt = (vecs_fre / np.sqrt(vals_fre)).dot(vecs_fre.T)
        eighs = [my_eigh(fre_inv_sqrt.dot(mat).dot(fre_inv_sqrt)) for mat in
        mats]

        # Log map of mats[n] at point fre is
        # sqrtm(fre).dot(logms[n]).dot(sqrtm(fre))
        logms = [(vecs * np.log(vals)).dot(vecs.T) for vals, vecs, success_flag
        in eighs]

        # Covariant derivative is fre.dot(logms_mean)
        logms_mean = np.mean(logms, axis=0)
        assert np.all(np.isfinite(logms_mean))

        # Norm of the covariant derivative on the tangent space at point fre
        norm_new = np.sqrt(np.trace(logms_mean.dot(logms_mean)))
        if tol is not None and norm_new < tol:
            tolerance_reached = True
            break

        if norm_new > norm_old:
            step = step / 2.
        else:
            vals_log, vecs_log, success_flag = my_eigh(logms_mean)

            # Move along the geodesic with stepsize step
            fre_sqrt = (vecs_fre * np.sqrt(vals_fre)).dot(vecs_fre.T)
            fre = fre_sqrt.dot(
            vecs_log * np.exp(vals_log * step)).dot(vecs_log.T).dot(fre_sqrt)

    if tol is not None and not tolerance_reached:
        warnings.warn("Maximum number of iterations reached without getting "
                      "to the requested tolerance level.")

    return fre


def grad_frechet_mean(mats, max_iter=10, tol=1e-3):
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
    # Positive definiteness check
    for mat in mats:
        assert_array_less(0.0, my_eigh(mat)[0])

    mats = my_stack(mats)

    # Initialization
    fre = np.mean(mats, axis=0)
    norm_old = np.inf
    step = 1.
    grad_norm = []
    for n in xrange(max_iter):
        vals_fre, vecs_fre, success_flag = my_eigh(fre)
        fre_inv_sqrt = (vecs_fre / np.sqrt(vals_fre)).dot(vecs_fre.T)
        eighs = [my_eigh(fre_inv_sqrt.dot(mat).dot(fre_inv_sqrt)) for mat in
        mats]

        # Log map of mats[n] at point fre is
        # sqrtm(fre).dot(logms[n]).dot(sqrtm(fre))
        logms = [(vecs * np.log(vals)).dot(vecs.T) for vals, vecs, success_flag
        in eighs]

        # Covariant derivative is fre.dot(logms_mean)
        logms_mean = np.mean(logms, axis=0)
        assert np.all(np.isfinite(logms_mean))

        # Norm of the covariant derivative on the tangent space at point fre
        norm_new = np.sqrt(np.trace(logms_mean.dot(logms_mean)))
        if tol is not None and norm_new < tol:
            grad_norm.append(norm_new)
            break

        if norm_new > norm_old:
            step = step / 2.
            grad_norm.append(norm_old)
        else:
            grad_norm.append(norm_new)
            vals_log, vecs_log, success_flag = my_eigh(logms_mean)

            # Move along the geodesic with stepsize step
            fre_sqrt = (vecs_fre * np.sqrt(vals_fre)).dot(vecs_fre.T)
            fre = fre_sqrt.dot(
            vecs_log * np.exp(vals_log * step)).dot(vecs_log.T).dot(fre_sqrt)

    return grad_norm


def random_diagonal_spd(shape):
    """Generates random positive definite diagonal matrix"""
    d = np.random.rand(shape, )
    return np.diag(d ** 2 + 1.)


def random_spd(shape):
    """Generates random symmetric positive definite matrix"""
    ran = np.random.rand(shape, shape)
    spd = ran.dot(ran.T) + np.eye(shape)  # spd, but possibly badly
                                          # conditionned
    vals, vecs, suc = my_eigh(spd)
    d = random_diagonal_spd(shape)
    return vecs.dot(d).dot(vecs.T)


def random_non_singular(shape):
    """Generates random non singular matrix"""
    d = random_diagonal_spd(shape)
    u = random_spd(shape)
    return linalg.inv(u).dot(d).dot(u)