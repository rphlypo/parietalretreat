import warnings
import random
import copy
from StringIO import StringIO

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_tuple_equal, assert_equal, \
    assert_is_instance, assert_raises_regexp
from scipy import linalg

import manifold as my_mfd


def random_diagonal(s, eig_min=0., eig_max=1.):
    """Generates random diagonal matrix, with diagonal elements in range
    [eig_min, eig_max].

    Parameters
    ==========
    s: int
        The first dimension of the array.
    eig_min: float, optional
        Lower bound for the diagonal elements.
    eig_max: float, optional
        Upper bound for the diagonal elements.

    Returns
    =======
    diag: array
        2D output diaogonal array, shape (s0, s0).
    """
    diag = np.random.rand(s) * (eig_max - eig_min) + eig_min
    return np.diag(diag)


def random_diagonal_spd(s, eig_min=1., eig_max=2.):
    """Generates random positive definite diagonal matrix"""
    assert(eig_min > 0)
    assert(eig_max > 0)
    return random_diagonal(s, eig_min, eig_max)


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


def test_inv():
    """Testing inv function"""
    m = random_spd(41)
    m_inv = my_mfd.inv(m)
    assert_array_almost_equal(m.dot(m_inv), np.eye(41))


def test_sqrtm():
    """Testing sqrtm function"""
    m = random_spd(12)
    m_sqrt = my_mfd.sqrtm(m)
    assert_array_almost_equal(m_sqrt.T, m_sqrt)
    assert_array_almost_equal(m_sqrt.dot(m_sqrt), m)


def test_expm():
    """Testing expm function"""
    m = np.random.rand(15, 15)
    m = m + m.T
    m_exp = my_mfd.expm(m)
    assert_array_almost_equal(m_exp.T, m_exp)
    assert_array_almost_equal(linalg.logm(m_exp), m)


def test_inv_sqrtm():
    """Testing inv_sqrtm function"""
    m = random_spd(21)
    m_inv_sqrt = my_mfd.inv_sqrtm(m)
    assert_array_almost_equal(m_inv_sqrt.T, m_inv_sqrt)
    assert_array_almost_equal(m_inv_sqrt.dot(m_inv_sqrt).dot(m), np.eye(21))


def test_my_stack():
    """Testing my_stack function"""
    mats = []
    for n in xrange(7):
        mats.append(np.random.rand(31, 31))
    stacked = my_mfd.my_stack(mats)
    assert_is_instance(stacked, np.ndarray)
    assert_tuple_equal(stacked.shape, (7, 31, 31))
    for n in xrange(7):
        assert_array_equal(stacked[n], mats[n])


def test_is_spd():
    """Testing is_spd function"""
    # Check error for non symmetric spd
    shape = random.randint(2, 50)
    mat_asym = np.random.rand(shape, shape)
    mat_non_spd = mat_asym + mat_asym.T
    mat_complex = np.asarray(mat_non_spd, dtype=complex)
    mat_complex[0, 0] = np.complex(0, random.uniform(1e-7, 10))
    mats = [mat_asym, mat_complex, mat_non_spd]
    err_outs = ["not symmetric", "non real", "negative eigenvalue"]
    for (mat, err_out) in zip(mats, err_outs):
        out = StringIO()
        my_mfd.is_spd(mat, out=out)
        output = out.getvalue().strip()
        assert(err_out in output)

    # Check no error for real spd
    mat = random_spd(shape)
    out = StringIO()
    assert(my_mfd.is_spd(mat, out=out))
    output = out.getvalue().strip()
    assert(output == "")


def geometric_mean_properties(non_singular, spds):
    """Testing geometric_mean function: generic, gradient descent and
    invariance properties.

    Parameters
    ==========
    non_singular: numpy.ndarray
        Non singular matrix.
    spds: list of numpy.ndarray
        List of spd matrices, same shape as non_singular.
    """
    input_spds = copy.copy(spds)
    geo = my_mfd.geometric_mean(spds)

    # Generic
    assert(isinstance(spds, list))
    for spd, input_spd in zip(spds, input_spds):
        assert_array_equal(spd, input_spd)
    assert_tuple_equal(geo.shape, spds[0].shape)
    assert(my_mfd.is_spd(geo))

    # Check warning if gradient norm in the last step is less than tolerance
    for (tol, max_iter) in [(1e-10, 1), (1e-7, 50)]:
        with warnings.catch_warnings(record=True) as w:
            geo = my_mfd.geometric_mean(spds, max_iter=max_iter,
                                                 tol=tol)
            grad_norm = my_mfd.grad_geometric_mean(spds, max_iter=max_iter,
                                                 tol=tol)
        if grad_norm[-1] > tol:
            assert_equal(len(grad_norm), max_iter)
            assert_equal(len(w), 1)
        else:
            assert_equal(len(w), 0)

    # Gradient norm is decreasing
    difference = np.diff(grad_norm)
    if np.any(difference):
        assert(not(np.amax(difference) > 0.))

    decimal = 5
    tol = 10 ** (-2 * decimal)
    max_iter = 100

    geo = my_mfd.geometric_mean(spds, tol=tol, max_iter=max_iter)

    # Invariance under reordering
    spds.reverse()
    if len(spds) > 1:
        spds.insert(0, spds[1])
        spds.pop(2)
    geo_new = my_mfd.geometric_mean(spds, tol=tol, max_iter=max_iter)
    assert_array_almost_equal(geo_new, geo)

    # Invariance under congruant transformation
    spds_cong = [non_singular.dot(spd).dot(non_singular.T) for spd in spds]
    geo_new = my_mfd.geometric_mean(spds_cong, tol=tol, max_iter=max_iter)
    assert_array_almost_equal(geo_new, non_singular.dot(geo).dot(
        non_singular.T))

    # Invariance under inversion
    spds_inv = [linalg.inv(spd) for spd in spds]
    geo_new = my_mfd.geometric_mean(spds_inv, tol=tol, max_iter=max_iter * 2)
    assert_array_almost_equal(geo_new, linalg.inv(geo), decimal=decimal)


def geometric_mean_exact(diags, couple, times, sym, non_singular):
    """Testing geometric_mean function: approximate mean close to the
    exact one.

    Parameters
    ==========
    diags: list of numpy.ndarrays
        List of diagonal spd matrices, same shape.
    couple: tuple, length 2
        couple of spd matrices, same shape.
    times: numpy.ndarray
        1D array of floats.
    sym: numpy.ndarray
        Symmetric matrix.
    non_singular: numpy.ndarray
        Non singular matrix, same shape as sym.
    """
    decimal = 7
    tol = 0.5 * 10 ** (-decimal)

    # Diagonal matrices
    exact_geo = np.prod(my_mfd.my_stack(diags), axis=0) ** \
                (1 / float(len(diags)))
    app_geo = my_mfd.geometric_mean(diags, max_iter=10, tol=tol)
    assert_array_almost_equal(app_geo, exact_geo, decimal)

    # 2 matrices
    A, B = couple
    B_sqrt = my_mfd.sqrtm(B)
    B_inv_sqrt = my_mfd.inv_sqrtm(B)
    exact_geo = B_sqrt.dot(
        my_mfd.sqrtm(B_inv_sqrt.dot(A).dot(B_inv_sqrt))).dot(B_sqrt)
    app_geo = my_mfd.geometric_mean([A, B], tol=tol)
    assert_array_almost_equal(app_geo, exact_geo, decimal)

    # Single geodesic matrices
    spds = []
    for time in times:
        spds.append(non_singular.dot(my_mfd.expm(time * sym)).dot(
            non_singular.T))
    exact_geo = non_singular.dot(my_mfd.expm(times.mean() * sym)).dot(
        non_singular.T)
    app_geo = my_mfd.geometric_mean(spds, tol=tol)
    assert_array_almost_equal(app_geo, exact_geo, decimal)


def test_geometric_loop():
    """Testing geometric mean function on multiple random enteries"""
    warnings.simplefilter('always', UserWarning)
    for n in [16, 60]:
        for shape in [15, 100]:
            # Check error for non spd enteries
            mat1 = np.ones((shape, shape))
            with assert_raises_regexp(ValueError, "at least one matrix is"):
                my_mfd.geometric_mean([mat1])

            # Generate input matrices
            spd1 = random_spd(shape)
            spd2 = random_spd(shape)
            spd_couple = (spd1, spd2)
            sym = random_spd(shape)
            sym = sym - 1.5 * np.eye(shape)
            times = np.empty(n, dtype=float)
            for k in xrange(n):
                times[k] = random.uniform(-20., -10.)
            non_singular = random_non_singular(shape)
            for p in [0, .25, .5, .75, 1]:  # proportion of badly conditionned
                                            # matrices
                diags = []
                spds = []
                for k in xrange(int(p * n)):
                    diags.append(random_diagonal_spd(shape, eig_min=1e-4,
                                                     eig_max=1e4))
                    spds.append(random_spd(shape, eig_min=1e-4, eig_max=1e4))
                for k in xrange(int(p * n), n):
                    diags.append(random_diagonal_spd(shape, eig_min=1.,
                                                     eig_max=10.))
                    spds.append(random_spd(shape, eig_min=1., eig_max=10.))

                # Tests
                geometric_mean_exact([diags[0]], spd_couple, times, sym,
                                     non_singular)
                geometric_mean_properties(non_singular, spds)