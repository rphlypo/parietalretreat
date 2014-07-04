import sys
import warnings
from StringIO import StringIO

import numpy as np
from scipy import linalg


def my_stack(arrays):
    """Stack arrays on the first axis.

    Parameters
    ==========
    arrays: list of numpy.ndarrays
        All arrays must have the same shape.

    Returns
    =======
    stacked: numpy.ndarray
        The array formed by stacking the given arrays.
    """
    stacked = np.concatenate([a[np.newaxis] for a in arrays])
    return stacked


def sqrtm(mat):
    """ Matrix square-root, for symetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array to be square rooted. Raise an error if the array is not
        square.

    Returns
    =======
    mat_sqrtm: (M, M) numpy.ndarray
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


def inv(mat):
    """ Inverse of matrix, for symmetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array to be inverted. Raise an error if the array is not square.

    Returns
    =======
    mat_inv: (M, M) numpy.ndarray
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


def inv_sqrtm(mat):
    """ Inverse of matrix square-root, for symetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array to be square rooted and inverted. Raise an error if the array
        is not square.

    Returns
    =======
    mat_inv_sqrtm: (M, M) numpy.ndarray
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


def logm(mat):
    """ Logarithm of matrix, for symetric positive definite matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array whose logarithm to be computed. Raise an error if the array is
        not square.

    Returns
    =======
    mat_logm: (M, M) numpy.ndarray
        Matrix logatrithm of mat.

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
    """ Exponential of matrix, for real symmetric matrices.

    Parameters
    ==========
    mat: (M, M) numpy.ndarray
        2D array whose exponential to be computed. Raise an error if the array is
        not square.

    Returns
    =======
    mat_exp: (M, M) numpy.ndarray
        Matrix exponential of mat.

    Note
    ====
    If input matrix is not real symmetric, no error is reported but results
    will be wrong.
    """
    if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
        raise ValueError('expected a square matrix')
    vals, vecs = linalg.eigh(mat)
    mat_exp = np.dot(vecs * np.exp(vals), vecs.T)
    return mat_exp


def is_spd(M, decimal=15, out=sys.stdout):
    """Assert that input matrix is symmetric positive definite.

    M must be symmetric down to specified decimal places and with no complex
    entry.
    The positive definiteness check is performed by checking that all
    eigenvalues are positive.

    Parameters
    ==========
    M: numpy.ndarray
        matrix.

    Returns
    =======
    is_spd: boolean
        True if matrix is symmetric positive definite, False otherwise.
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


def geometric_mean(mats, max_iter=10, tol=1e-7):
    """ Computes the geometric mean of a list of symmetric positive definite
    matrices.

    Minimization of the objective function by an intrinsic gradient descent in
    the manifold: moving from the current point geo to the next one is
    done along a short geodesic arc in the opposite direction of the covariant
    derivative of the objective function evaluated at point geo.

    See Algorithm 3 of:
        P. Thomas Fletcher, Sarang Joshi. Riemannian Geometry for the
        Statistical Analysis of Diffusion Tensor Data. Signal Processing, 2007.

    Parameters
    ==========
    mats: list of numpy.array
        list of symmetric positive definite matrices, same shape.
    max_iter: int, optional
        maximal number of iterations.
    tol: float, optional
        tolerance.

    Returns
    =======
    geo: numpy.array
        Geometric mean of the matrices.
    """
    # Shape and symmetry positive definiteness checks
    for mat in mats:
        out = StringIO()
        if mat.ndim != 2 or mat.shape[0] != mat.shape[-1]:
            raise ValueError('at least one array is not square')
        if not is_spd(mat, out=out):
            output = out.getvalue().strip()
            raise ValueError("at least one matrix is not real spd:" + output)

    # Initialization
    mats = my_stack(mats)
    geo = np.mean(mats, axis=0)
    tolerance_reached = False
    norm_old = np.inf
    step = 1.

    # Gradient descent
    for n in xrange(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_inv_sqrt = (vecs_geo / np.sqrt(vals_geo)).dot(vecs_geo.T)
        eighs = [linalg.eigh(geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)) for
                 mat in mats]
        logs = [(vecs * np.log(vals)).dot(vecs.T) for vals, vecs in eighs]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - geo.dot(logms_mean)
        try:
            assert np.all(np.isfinite(logs_mean))
        except AssertionError:
            raise FloatingPointError("Nan value after logarithm operation")
        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point geo

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        geo_sqrt = (vecs_geo * np.sqrt(vals_geo)).dot(vecs_geo.T)
        geo = geo_sqrt.dot(vecs_log * np.exp(vals_log * step)).dot(
            vecs_log.T).dot(geo_sqrt)  # Move along the geodesic with step size
                                       # step

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            step = step / 2.
            norm = norm_old
        if tol is not None and norm / geo.size < tol:
            tolerance_reached = True
            break

    if tol is not None and not tolerance_reached:
        warnings.warn("Maximum number of iterations reached without" +\
                      " getting to the requested tolerance level.")

    return geo


def grad_geometric_mean(mats, max_iter=10, tol=1e-7):
    """ Returns at each iteration step of the geometric_mean algorithm the norm
    of the covariant derivative. Norm is intrinsic norm on the tangent space at
    the geometric mean at the current step.

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
    mats = my_stack(mats)

    # Initialization
    geo = np.mean(mats, axis=0)
    norm_old = np.inf
    step = 1.
    grad_norm = []
    for n in xrange(max_iter):
        # Computation of the gradient
        vals_geo, vecs_geo = linalg.eigh(geo)
        geo_inv_sqrt = (vecs_geo / np.sqrt(vals_geo)).dot(vecs_geo.T)
        eighs = [linalg.eigh(geo_inv_sqrt.dot(mat).dot(geo_inv_sqrt)) for
                 mat in mats]
        logs = [(vecs * np.log(vals)).dot(vecs.T) for vals, vecs in eighs]
        logs_mean = np.mean(logs, axis=0)  # Covariant derivative is
                                           # - geo.dot(logms_mean)
        try:
            assert np.all(np.isfinite(logs_mean))
        except AssertionError:
            raise FloatingPointError("Nan value after logarithm operation")
        norm = np.linalg.norm(logs_mean)  # Norm of the covariant derivative on
                                          # the tangent space at point geo

        # Update of the minimizer
        vals_log, vecs_log = linalg.eigh(logs_mean)
        geo_sqrt = (vecs_geo * np.sqrt(vals_geo)).dot(vecs_geo.T)
        geo = geo_sqrt.dot(vecs_log * np.exp(vals_log * step)).dot(
            vecs_log.T).dot(geo_sqrt)  # Move along the geodesic with step size
                                       # step

        # Update the norm and the step size
        if norm < norm_old:
            norm_old = norm
        if norm > norm_old:
            step = step / 2.
            norm = norm_old

        grad_norm.append(norm / geo.size)
        if tol is not None and norm / geo.size < tol:
            break

    return grad_norm