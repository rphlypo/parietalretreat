import nose
import warnings
import random
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less, \
    assert_array_equal
from nose.tools import assert_not_equal, assert_tuple_equal, assert_equal, \
    assert_is_instance, assert_almost_equal, assert_raises_regexp
from scipy import linalg

import manifold as my_mfd


def test_random_diagonal_spd():
    """Testing random_diagonal_spd function"""
    d = my_mfd.random_diagonal_spd(15)
    diag = np.diag(d)
    assert_array_almost_equal(d, np.diag(diag))
    assert_array_less(0.0, diag)


def test_random_spd():
    """Testing random_spd function"""
    spd = my_mfd.random_spd(17)
    assert_array_almost_equal(spd, spd.T)
    assert(np.all(np.isreal(spd)))
    assert_array_less(0., np.linalg.eigvalsh(spd))


def test_random_non_singular():
    """Testing random_non_singular function"""
    non_sing = my_mfd.random_non_singular(23)
    assert_not_equal(linalg.det(non_sing), 0.)


def test_inv():
    """Testing inv function"""
    m = my_mfd.random_spd(41)
    m_inv = my_mfd.inv(m)
    assert_array_almost_equal(m.dot(m_inv), np.eye(41))


def test_sqrtm():
    """Testing sqrtm function"""
    m = my_mfd.random_spd(12)
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
    m = my_mfd.random_spd(21)
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
    spd = my_mfd.random_spd(19)
    vals, vecs = linalg.eigh(spd)
    assert_almost_equal(abs(linalg.det(vecs)), 1.)
    assert_array_almost_equal(vecs.dot(vecs.T), np.eye(19))
    assert_array_almost_equal((vecs * vals).dot(vecs.T), spd)


def test_covariant_derivative():
    """Testing covariant_derivative function"""


def test_frechet_mean():  # TODO: test suite and split in several tests
    """Testing frechet_mean function"""
    n = random.randint(3, 50)
    shape = random.randint(1, 50)
    spds = []
    for k in xrange(n):
        eig_min = random.uniform(1e-7, 1.)
        eig_max = random.uniform(1., 1e7)
        spds.append(my_mfd.random_spd(shape, eig_min, eig_max))
    fre = my_mfd.frechet_mean(spds)

    # Generic
    assert_tuple_equal(fre.shape, spds[0].shape)
    assert_array_almost_equal(fre, fre.T)
    assert_array_less(0., linalg.eigvals(fre))

    # Check error for non spd enteries
    mat1 = np.ones((shape, shape))
    with assert_raises_regexp(ValueError, "at least one matrix is not real " +\
    "spd"):
        my_mfd.frechet_mean([mat1])

    # Check warning if gradient norm in the last step is less than tolerance
    for (tol, max_iter) in [(1e-10, 1), (1e-3, 50)]:
        with warnings.catch_warnings(record=True) as w:
            fre = my_mfd.frechet_mean(spds, max_iter=max_iter, tol=tol)
            grad_norm = my_mfd.grad_frechet_mean(spds, max_iter=max_iter,
                                                  tol=tol)
            if grad_norm[-1] > tol:
                assert_equal(len(w), 2)
                assert_equal(len(grad_norm), max_iter)

    # Gradient norm is decreasing
    assert_array_less(np.diff(grad_norm), 0.)

    decimal = 5
    tol = 10 ** (-2 * decimal)
    fre = my_mfd.frechet_mean(spds, tol=tol)

    # Adaptative version requires less iterations than non adaptaive
    grad_norm = my_mfd.grad_frechet_mean(spds, tol=tol)
    grad_norm_fast = my_mfd.grad_frechet_mean(spds, tol=tol, adaptative=True)
    assert(len(grad_norm_fast) == len(grad_norm))

    # Invariance under reordering
    spds.reverse()
    spds.insert(0, spds[2])
    spds.pop(3)
    fre_new = my_mfd.frechet_mean(spds, tol=tol)
    assert_array_almost_equal(fre_new, fre)

    # Invariance under congruant transformation
    c = my_mfd.random_non_singular(shape)
    spds_cong = [c.dot(spd).dot(c.T) for spd in spds]
    fre_new = my_mfd.frechet_mean(spds_cong, tol=tol)
    assert_array_almost_equal(fre_new, c.dot(fre).dot(c.T))

    # Invariance under inversion
    spds_inv = [linalg.inv(spd) for spd in spds]
    fre_new = my_mfd.frechet_mean(spds_inv, tol=tol)
    assert_array_almost_equal(fre_new, linalg.inv(fre), decimal=decimal)

    # Approximate Frechet mean is close to the exact one
    decimal = 7
    tol = 0.5 * 10 ** (-decimal)

    # Diagonal matrices: exact Frechet mean is geometric mean
    diags = []
    for k in xrange(n):
        diags.append(my_mfd.random_diagonal_spd(shape, 1e-5, 1e5))
    exact_fre = np.prod(my_mfd.my_stack(diags), axis=0) ** \
                (1 / float(len(diags)))
    app_fre = my_mfd.frechet_mean(diags, max_iter=10, tol=tol)
    assert_array_almost_equal(app_fre, exact_fre, decimal)

    # 2 matrices
    spd1 = my_mfd.random_spd(shape)
    spd2 = my_mfd.random_spd(shape)
    spd2_sqrt = my_mfd.sqrtm(spd2)
    spd2_inv_sqrt = my_mfd.inv_sqrtm(spd2)
    exact_fre = spd2_sqrt.dot(
    my_mfd.sqrtm(spd2_inv_sqrt.dot(spd1).dot(spd2_inv_sqrt))).dot(spd2_sqrt)
    app_fre = my_mfd.frechet_mean([spd1, spd2], tol=tol)
    assert_array_almost_equal(app_fre, exact_fre, decimal)

    # Single geodesic matrices : TODO
    S = np.random.rand(shape, shape)
    S = S + S.T
    times = np.empty(n, dtype=float)
    spds = []
    for k in xrange(n):
        t_k = random.uniform(-2., -1.)
        times[k] = t_k
        spds.append(c.dot(my_mfd.expm(t_k * S)).dot(c.T))
#        assert_array_almost_equal(my_mfd.expm(S), my_mfd.expm(S).T)
#    exact_fre = c.dot(my_mfd.expm(times.mean() * S)).dot(c.T)
#    app_fre = my_mfd.frechet_mean(spds, tol=tol)
#    assert_array_almost_equal(app_fre, exact_fre, decimal)

if __name__ == "__main__":
    nose.run()