import copy

import numpy as np
from scipy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises, assert_equal, assert_is_instance,\
    assert_true

import connectivity as my_con
from manifold import is_spd, sqrtm, expm
from test_manifold import random_spd


def test_sym_to_vec():
    """Testing sym_to_vec function"""
    # Check output value is correct
    sym = np.ones((3, 3))
    vec = my_con.sym_to_vec(sym)
    vec_expected = np.array([1., np.sqrt(2), 1., np.sqrt(2),  np.sqrt(2), 1.])
    vec_bool = my_con.sym_to_vec(sym > 0, isometry=False)
    bool_expected = np.ones(6, dtype=bool)
    assert_array_almost_equal(vec, vec_expected)
    assert_array_equal(vec_bool, bool_expected)

    # Check vec_to_sym is the inverse function of sym_to_vec
    shape = 19
    m = np.random.rand(shape, shape)
    sym = m + m.T
    syms = np.asarray([sym, 2. * sym, 0.5 * sym])
    vec = my_con.sym_to_vec(sym)
    vecs = my_con.sym_to_vec(syms)
    assert_array_almost_equal(my_con.vec_to_sym(vec), sym)
    assert_array_almost_equal(my_con.vec_to_sym(vecs), syms)
    vec = my_con.sym_to_vec(sym, isometry=False)
    vecs = my_con.sym_to_vec(syms, isometry=False)
    assert_array_almost_equal(my_con.vec_to_sym(vec, isometry=False), sym)
    assert_array_almost_equal(vec[..., -shape:], sym[..., -1, :])
    assert_array_almost_equal(
    my_con.vec_to_sym(vecs, isometry=False), syms)
    assert_array_almost_equal(vecs[..., -shape:], syms[..., -1, :])


def test_vec_to_sym():
    """Testing vec_to_sym function"""
    # Check error if unsuitable size
    vec = np.random.rand(31)
    with assert_raises(ValueError) as ve:
        my_con.vec_to_sym(vec)
        assert_equal(len(ve), 1)

    # Check output value is correct
    vec = np.ones(6, )
    sym = my_con.vec_to_sym(vec)
    sym_expected = np.array([[np.sqrt(2), 1., 1.], [1., np.sqrt(2), 1.],
                              [1., 1., np.sqrt(2)]]) / np.sqrt(2)
    mask = my_con.vec_to_sym(vec > 0, isometry=False)
    mask_expected = np.ones((3, 3), dtype=bool)
    assert_array_almost_equal(sym, sym_expected)
    assert_array_equal(mask, mask_expected)

    # Check sym_to_vec the inverse function of vec_to_sym
    n = 41
    p = n * (n + 1) / 2
    vec = np.random.rand(p)
    sym = my_con.vec_to_sym(vec)
    assert_array_almost_equal(my_con.sym_to_vec(sym), vec)
    sym = my_con.vec_to_sym(vec, isometry=False)
    assert_array_almost_equal(my_con.sym_to_vec(sym, isometry=False), vec)
    vecs = np.asarray([vec, 2. * vec, 0.5 * vec])
    syms = my_con.vec_to_sym(vecs)
    assert_array_almost_equal(my_con.sym_to_vec(syms), vecs)
    syms = my_con.vec_to_sym(vecs, isometry=False)
    assert_array_almost_equal(my_con.sym_to_vec(syms, isometry=False), vecs)


def test_prec_to_partial():
    """Testing prec_to_partial function"""
    shape = 101
    prec = random_spd(shape)
    partial = my_con.prec_to_partial(prec)
    assert_true(is_spd(partial))
    d = np.sqrt(np.diag(np.diag(prec)))
    assert_array_almost_equal(
    d.dot(partial).dot(d), -prec + 2 * np.diag(np.diag(prec)))


def test_transform():
    """Testing fit_transform method for class CovEmbedding"""
    n_subjects = 49
    shape = 95
    n_samples = 300

    # Generate signals and compute empirical covariances
    covs = []
    signals = []
    for k in xrange(n_subjects):
        signal = np.random.randn(n_samples, shape)
        signals.append(signal)
        signal -= signal.mean(axis=0)
        covs.append((signal.T).dot(signal) / n_samples)

    input_covs = copy.copy(covs)
    for kind in ["correlation", "tangent", "precision", "partial correlation"]:
        estimators = {'kind': kind, 'cov_estimator': None}
        cov_embedding = my_con.CovEmbedding(**estimators)
        covs_transformed = cov_embedding.fit_transform(signals)

        # Generic
        assert_is_instance(covs_transformed, np.ndarray)
        assert_equal(len(covs_transformed), len(covs))

        for k, vec in enumerate(covs_transformed):
            assert_equal(vec.size, shape * (shape + 1) / 2)
            assert_array_equal(input_covs[k], covs[k])
            cov_new = my_con.vec_to_sym(vec)
            assert_true(is_spd(covs[k]))

            # Positive definiteness if expected and output value checks
            if estimators["kind"] == "tangent":
                assert_array_almost_equal(cov_new, cov_new.T)
                fre_sqrt = sqrtm(cov_embedding.mean_cov_)
                assert_true(is_spd(fre_sqrt))
                assert_true(is_spd(cov_embedding.whitening_))
                assert_array_almost_equal(
                cov_embedding.whitening_.dot(fre_sqrt), np.eye(shape))
                assert_array_almost_equal(
                fre_sqrt.dot(expm(cov_new)).dot(fre_sqrt), covs[k])
            if estimators["kind"] == "precision":
                assert_true(is_spd(cov_new))
                assert_array_almost_equal(cov_new.dot(covs[k]), np.eye(shape))
            if estimators["kind"] == "correlation":
                assert_true(is_spd(cov_new))
                d = np.sqrt(np.diag(np.diag(covs[k])))
                assert_array_almost_equal(d.dot(cov_new).dot(d), covs[k])
            if estimators["kind"] == "partial correlation":
                prec = linalg.inv(covs[k])
                d = np.sqrt(np.diag(np.diag(prec)))
                assert_array_almost_equal(
                d.dot(cov_new).dot(d), -prec + 2 * np.diag(np.diag(prec)))