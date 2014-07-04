import sys
import warnings
from StringIO import StringIO

import numpy as np
from numpy.testing import assert_array_almost_equal
from scipy import linalg


def my_stack(arrays):
    """Stack arrays on the first axis.

    Parameters
    ==========
    arrays: list of ndarrays
        All arrays must have the same shape, except in the first dimension.

    Returns
    =======
    stacked: ndarray
        The array formed by stacking the given arrays.
    """
    stacked = np.concatenate([a[np.newaxis] for a in arrays])
    return stacked


def sqrtm(mat):
    """ Matrix square-root, for symetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) ndarray
        2D array to be squared. Raise an error

    Returns
    =======
    mat_sqrtm: (M, M) ndarray
        The symmetric matrix square root of mat.

    Note
    ====
    If input matrix is not symmetric positive definite, no error is reported
    but results will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_sqrtm = np.dot(vecs * np.sqrt(vals), vecs.T)
    return mat_sqrtm


def inv_sqrtm(mat):
    """ Inverse of matrix square-root, for symetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) ndarray
        2D array to be squared.

    Returns
    =======
    mat_inv_sqrtm: (M, M) ndarray
        The inverse matrix of the symmetric square root of mat.

    Note
    ====
    If input matrix is not symmetric positive definite, no error is reported
    but results will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_inv_sqrtm = np.dot(vecs / np.sqrt(vals), vecs.T)
    return mat_inv_sqrtm


def inv(mat):
    """ Inverse of matrix, for symmetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) ndarray
        2D array to be squared.

    Returns
    =======
    mat_inv: (M, M) ndarray
        The inverse matrix of mat.

    Note
    ====
    If input matrix is not symmetric positive definite, no error is reported
    but results will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_inv = np.dot(vecs / vals, vecs.T)
    return mat_inv


def logm(mat):
    """ Logarithm of matrix, for symetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) ndarray
        2D array to be squared.

    Returns
    =======
    mat_logm: (M, M) ndarray
        The inverse matrix of mat.

    Note
    ====
    If input matrix is not symmetric positive definite, no error is reported
    but results will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_logm = np.dot(vecs * np.log(vals), vecs.T)
    return mat_logm


def expm(mat):
    """ Exponential of matrix, for real symmetric matrices
    """
    try:
        assert_array_almost_equal(mat, mat.T)
        assert(np.all(np.isreal(mat)))
    except AssertionError:
        raise ValueError("matrix is not real symmetric")
    vals, vecs = linalg.eigh(mat)
    return np.dot(vecs * np.exp(vals), vecs.T)


def is_spd(M, decimal=15, out=sys.stdout):
    """Assert that input matrix is real symmetric positive definite.

    M must be symmetric down to specified decimal places and with no complex
    entry.output = out.getvalue().strip()
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
    if np.any(np.isnan(M)) or np.any(np.isinf(M)):
        out.write("matrix has nan or inf entery")
        return False
    if not np.allclose(M, M.T, atol=0.1 ** decimal):
        out.write("matrix not symmetric to {0} decimals".format(decimal))
        return False
    if np.any(np.iscomplex(M)):
        out.write("matrix has a non real value {0}".format(
            M[np.iscomplex(M)][0]))
        return False
    eigvalsh = np.linalg.eigvalsh(M)
    ispd = eigvalsh.min() > 0
    if not ispd:
        out.write("matrix has a negative eigenvalue: {0:.3f}".format(
            eigvalsh.min()))
    return ispd


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
        out = StringIO()
        if not is_spd(mat, out=out):
            output = out.getvalue().strip()
            raise ValueError("at least one matrix is not real spd:" + output)

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

        # Norm of the covariant derivative on the tangent space at point fre
        norm = np.sqrt(np.trace(logms_mean.dot(logms_mean)))
        if norm > norm_old:
            step = step / 2.
            norm = norm_old
        if norm < norm_old:
            norm_old = norm
            if adaptative:
                step = 2. * step
        if tol is not None and norm < tol:
            tolerance_reached = True
            break

        # Move along the geodesic with stepsize step
        fre_sqrt = (vecs_fre * np.sqrt(vals_fre)).dot(vecs_fre.T)
        fre = fre_sqrt.dot(
            vecs_log * np.exp(vals_log * step)).dot(vecs_log.T).dot(fre_sqrt)

    if tol is not None and not tolerance_reached:
        warnings.warn("Maximum number of iterations reached without" +\
                      " getting to the requested tolerance level.")

    return fre


def grad_frechet_mean(mats, max_iter=10, tol=1e-3, adaptative=False):
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
        out = StringIO()
        if not is_spd(mat, out=out):
            output = out.getvalue().strip()
            raise ValueError("at least one matrix is not real spd:" + output)

    mats = my_stack(mats)

    # Initialization
    fre = np.mean(mats, axis=0)
    tolerance_reached = False
    norm_old = np.inf
    step = 1.
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

        # Norm of the covariant derivative on the tangent space at point fre
        norm = np.sqrt(np.trace(logms_mean.dot(logms_mean)))
        if norm > norm_old:
            step = step / 2.
            norm = norm_old
        if norm < norm_old:
            norm_old = norm
            if adaptative:
                step = 2. * step
        grad_norm.append(norm)
        if tol is not None and norm < tol:
            tolerance_reached = True
            break

        # Move along the geodesic with stepsize step
        fre_sqrt = (vecs_fre * np.sqrt(vals_fre)).dot(vecs_fre.T)
        fre = fre_sqrt.dot(
            vecs_log * np.exp(vals_log * step)).dot(vecs_log.T).dot(fre_sqrt)

    if tol is not None and not tolerance_reached:
        print "$$$$$$$$$$ generating warning $$$$$$$$$$$$$$$$$"
        warnings.warn("Maximum number of iterations reached without" +\
                      " getting to the requested tolerance level.")

    return grad_norm