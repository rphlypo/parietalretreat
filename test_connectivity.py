# Standard library imports
import random

# Related third party imports
import nose
import numpy as np
from scipy import linalg
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises, assert_equal, assert_is_instance,\
    assert_true

# Local application/library specific imports
import connectivity as my_con
import manifold as my_mfd


def test_sym_to_vec():
    """Testing sym_to_vec function"""
    sym = np.ones((3, 3))
    vec = my_con.sym_to_vec(sym)
    vec_expected = np.array([1., np.sqrt(2), 1., np.sqrt(2),  np.sqrt(2), 1.])
    vec_bool = my_con.sym_to_vec(sym > 0, isometry=False)
    bool_expected = np.ones(6, dtype=bool)
    assert_array_almost_equal(vec, vec_expected)
    assert_array_equal(vec_bool, bool_expected)

    shape = random.randint(1, 40)
    m = np.random.rand(shape, shape)
    sym = m + m.T
    syms = np.asarray([sym, 2. * sym, 0.5 * sym])
    vec = my_con.sym_to_vec(sym)
    vecs = my_con.sym_to_vec(syms)
    assert_array_almost_equal(my_con.vec_to_sym(vec), sym)
    for k, vec in enumerate(vecs):
        assert_array_almost_equal(my_con.vec_to_sym(vec), syms[k])
    vec = my_con.sym_to_vec(sym, isometry=False)
    vecs = my_con.sym_to_vec(syms, isometry=False)
    assert_array_almost_equal(my_con.vec_to_sym(vec, isometry=False), sym)
    assert_array_almost_equal(vec[..., -shape:], sym[..., -1, :])
    for k, vec in enumerate(vecs):
        assert_array_almost_equal(
        my_con.vec_to_sym(vec, isometry=False), syms[k])
    assert_array_almost_equal(vecs[..., -shape:], syms[..., -1, :])


def test_vec_to_sym():
    """Testing vec_to_sym function"""
    # Check error if unsuitable size
    vec = np.random.rand(31)
    with assert_raises(ValueError) as ve:
        my_con.vec_to_sym(vec)
        assert_equal(len(ve), 1)

    # Test for random suitable size
    n = random.randint(1, 50)
    p = n * (n + 1) / 2
    vec = np.random.rand(p)
    sym = my_con.vec_to_sym(vec)
    assert_array_almost_equal(my_con.sym_to_vec(sym), vec)

    vec = np.ones(6, )
    sym = my_con.vec_to_sym(vec)
    sym_expected = np.array([[np.sqrt(2), 1., 1.], [1., np.sqrt(2), 1.],
                              [1., 1., np.sqrt(2)]]) / np.sqrt(2)
    sym_bool = my_con.vec_to_sym(vec > 0, isometry=False)
    bool_expected = np.ones((3, 3), dtype=bool)
    assert_array_almost_equal(sym, sym_expected)
    assert_array_equal(sym_bool, bool_expected)


def test_prec_to_partial():
    """Testing prec_to_partial function"""
    shape = random.randint(1, 50)
    prec = my_mfd.random_spd(shape)
    partial = my_con.prec_to_partial(prec)
    assert_true(my_mfd.is_spd(partial))
    d = np.sqrt(np.diag(np.diag(prec)))
    assert_array_almost_equal(
    d.dot(partial).dot(d), -prec + 2 * np.diag(np.diag(prec)))

def test_transform():  # TODO : class test for class CovEmbedding
    """Testing fit_transform method for class CovEmbedding"""
    n_subjects = random.randint(3, 50)
    shape = random.randint(1, 10)
    n_samples = 300
    covs = []
    signals = []
    for k in xrange(n_subjects):
        signal = np.random.randn(n_samples, shape)
        signals.append(signal)
        signal -= signal.mean(axis=0)
        covs.append((signal.T).dot(signal) / n_samples)
    for kind in ["correlation", "precision", "partial correlation", "tangent"]:
        estimators = {'kind': kind, 'cov_estimator': None}
        cov_embedding = my_con.CovEmbedding(**estimators)
        covs_transformed = cov_embedding.fit_transform(signals)

        # Generic
        assert_is_instance(covs_transformed, np.ndarray)
        assert_equal(len(covs_transformed), len(covs))

        for k, vec in enumerate(covs_transformed):
            assert_equal(vec.size, shape * (shape + 1) / 2)
            cov_new = my_con.vec_to_sym(vec)
            assert_true(my_mfd.is_spd(covs[k]))
            if estimators["kind"] == "tangent":
                assert_array_almost_equal(cov_new, cov_new.T)
                fre_sqrt = my_mfd.sqrtm(cov_embedding.mean_cov_)
                assert_true(my_mfd.is_spd(fre_sqrt))
                assert_true(my_mfd.is_spd(cov_embedding.whitening_))
                assert_array_almost_equal(
                cov_embedding.whitening_.dot(fre_sqrt), np.eye(shape))
                assert_array_almost_equal(
                fre_sqrt.dot(my_mfd.expm(cov_new)).dot(fre_sqrt), covs[k])
            if estimators["kind"] == "precision":
                assert_true(my_mfd.is_spd(cov_new))
                assert_array_almost_equal(cov_new.dot(covs[k]), np.eye(shape))
            if estimators["kind"] == "correlation":
                assert_true(my_mfd.is_spd(cov_new))
                d = np.sqrt(np.diag(np.diag(covs[k])))
                assert_array_almost_equal(d.dot(cov_new).dot(d), covs[k])
            if estimators["kind"] == "partial correlation":
                assert_true(my_mfd.is_spd(cov_new))
                prec = linalg.inv(covs[k])
                d = np.sqrt(np.diag(np.diag(prec)))
                assert_array_almost_equal(
                d.dot(cov_new).dot(d), -prec + 2 * np.diag(np.diag(prec)))


if __name__ == "__main__":
    nose.run()