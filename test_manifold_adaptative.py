import nose
import warnings
import random
import copy
from StringIO import StringIO

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less, \
    assert_array_equal
from nose.tools import assert_tuple_equal, assert_equal, \
    assert_is_instance, assert_almost_equal, assert_raises_regexp
from scipy import linalg
from scipy.stats import gmean
from matrices_generator import random_spd, random_diagonal_spd,\
    random_non_singular

import manifold as my_mfd


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


def test_my_eigh():
    """Testing my_eigh function"""
    spd = random_spd(19)
    vals, vecs = linalg.eigh(spd)
    assert_almost_equal(abs(linalg.det(vecs)), 1.)
    assert_array_almost_equal(vecs.dot(vecs.T), np.eye(19))
    assert_array_almost_equal((vecs * vals).dot(vecs.T), spd)


def test_is_spd():
    """Testing is_spd function"""
    # Check error for non real symmetric spd
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


def frechet_mean_properties(non_singular, spds):
    """Testing frechet_mean function: generic, gradient descent and invariance
    properties.

    Parameters
    ----------
    non_singular: array
        non singular matrix
    spds: list of array
        list of spd matrices, same shape as non_singular
    """
    input_spds = copy.copy(spds)
    fre = my_mfd.frechet_mean(spds)

    # Generic
    assert(isinstance(spds, list))
    for spd, input_spd in zip(spds, input_spds):
        assert_array_equal(spd, input_spd)
    assert_tuple_equal(fre.shape, spds[0].shape)
    assert_array_almost_equal(fre, fre.T)
    assert_array_less(0., linalg.eigvals(fre))

    # Check warning if gradient norm in the last step is less than tolerance
    for (tol, max_iter) in [(1e-10, 1), (1e-3, 50)]:
        with warnings.catch_warnings(record=True) as w:
            fre = my_mfd.frechet_mean(spds, max_iter=max_iter,
                                                 tol=tol)
            grad_norm = my_mfd.grad_frechet_mean(spds, max_iter=max_iter,
                                                 tol=tol)
        if grad_norm[-1] > tol:
            assert_equal(len(grad_norm), max_iter)
            assert_equal(len(w), 2)

    # Gradient norm is decreasing
#    assert((np.amax(np.diff(grad_norm)) == 0.) or\
#        (np.amax(np.diff(grad_norm)) < 0.))
    decimal = 5
    tol = 10 ** (-2 * decimal)
    max_iter = 100

    # Adaptative version requires more iterations than non adaptative!
    grad_norm = my_mfd.grad_frechet_mean(spds, tol=tol, max_iter=max_iter)
    grad_norm_fast = my_mfd.grad_frechet_mean(spds, tol=tol, max_iter=max_iter,
                                              adaptative=True)
#    assert(len(grad_norm_fast) >= len(grad_norm))

    # Conditin number of Frechet mean is less than the geometric mean of
    # condition numbers
    conds = [np.linalg.cond(spd) for spd in spds]
    assert(np.linalg.cond(fre) <= gmean(conds))
    eucl_mean = my_mfd.frechet_mean(spds, tol=1e5, max_iter=1)
    print np.linalg.cond(fre)
    print np.linalg.cond(eucl_mean)
    print np.max(conds), np.min(conds)
    assert(np.linalg.cond(fre) <= 2 * np.linalg.cond(eucl_mean))

    fre = my_mfd.frechet_mean(spds, tol=tol, max_iter=max_iter)

    # Invariance under reordering
    spds.reverse()
    spds.insert(0, spds[2])
    spds.pop(3)
    fre_new = my_mfd.frechet_mean(spds, tol=tol, max_iter=max_iter)
    assert_array_almost_equal(fre_new, fre)

    # Invariance under congruant transformation
    spds_cong = [non_singular.dot(spd).dot(non_singular.T) for spd in spds]
    fre_new = my_mfd.frechet_mean(spds_cong, tol=tol, max_iter=max_iter)
    assert_array_almost_equal(fre_new, non_singular.dot(fre).dot(
        non_singular.T))

    # Invariance under inversion
    spds_inv = [linalg.inv(spd) for spd in spds]
    fre_new = my_mfd.frechet_mean(spds_inv, tol=tol, max_iter=max_iter * 2)
    assert_array_almost_equal(fre_new, linalg.inv(fre), decimal=decimal)


def frechet_mean_exact(diags, couple, times, sym, non_singular):
    """Testing frechet_mean function: approximate mean close to the
    exact one.

    Parameters
    ----------
    diags: list of arrays
        list of diagonal spd matrices, same shape
    couple: tuple, length 2
        couple of spd matrices, same shape
    times: array
        1d array of floats
    sym: array
        symmetric matrix
    non_singular: array
        non singular matrix, same shape as sym
    """
    decimal = 7
    tol = 0.5 * 10 ** (-decimal)

    # Diagonal matrices: exact Frechet mean is geometric mean
    exact_fre = np.prod(my_mfd.my_stack(diags), axis=0) ** \
                (1 / float(len(diags)))
    app_fre = my_mfd.frechet_mean(diags, max_iter=10, tol=tol)
    assert_array_almost_equal(app_fre, exact_fre, decimal)

    # 2 matrices
    A, B = couple
    B_sqrt = my_mfd.sqrtm(B)
    B_inv_sqrt = my_mfd.inv_sqrtm(B)
    exact_fre = B_sqrt.dot(
    my_mfd.sqrtm(B_inv_sqrt.dot(A).dot(B_inv_sqrt))).dot(B_sqrt)
    app_fre = my_mfd.frechet_mean([A, B], tol=tol)
    assert_array_almost_equal(app_fre, exact_fre, decimal)

    # Single geodesic matrices
    spds = []
    for time in times:
        spds.append(non_singular.dot(my_mfd.expm(time * sym)).dot(
            non_singular.T))
    exact_fre = non_singular.dot(my_mfd.expm(times.mean() * sym)).dot(
        non_singular.T)
    app_fre = my_mfd.frechet_mean(spds, tol=tol)
    assert_array_almost_equal(app_fre, exact_fre, decimal)


def test_frechet_loop():
    """Testing Frechet mean function on multiple random enteries"""
    warnings.simplefilter('always', UserWarning)
    for n in xrange(3, 10, 4):
        print "number of matrices = {0}".format(n)
        for shape in xrange(2, 100, 50):
            print "shape of each matrix = ({0}, {0})".format(shape)
            # Check error for non spd enteries
            mat1 = np.ones((shape, shape))
            with assert_raises_regexp(ValueError, "at least one matrix is"):
                my_mfd.frechet_mean([mat1])

            # Generate input matrices
            for eig_decimal in xrange(1, 7, 2):
                print "eigenvalues range from {0} to {1}".format(
                    1. / 10 ** eig_decimal, 10 ** eig_decimal)
                spds = []
                for k in xrange(n):
                    eig_min = random.uniform(1. / 10 ** eig_decimal, 1.)
                    eig_max = random.uniform(1., 10 ** eig_decimal)
                    spds.append(random_spd(shape, eig_min, eig_max))

                diags = []
                for k in xrange(n):
                    diags.append(random_diagonal_spd(shape, 1e-5, 1e5))

                spd1 = random_spd(shape)
                spd2 = random_spd(shape)
                couple = (spd1, spd2)

                sym = random_spd(shape)
                sym = sym - 1.5 * np.eye(shape)
                times = np.empty(n, dtype=float)
                for k in xrange(n):
                    times[k] = random.uniform(-20., -10.)

                non_singular = random_non_singular(shape)

                # Tests
                frechet_mean_properties(non_singular, spds)
                frechet_mean_exact(diags, couple, times, sym, non_singular)
    print stop


if __name__ == "__main__":
    nose.run()