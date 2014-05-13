import copy

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EmpiricalCovariance
import manifold as spd_mfd


def sym_to_vec(sym, isometry=True):
    """ Returns the lower triangular part of
    sqrt(2) sym + (1-sqrt(2)) * diag(sym),

    sqrt(2) * offdiag(sym) + np.diag(np.diag(sym))

    shape n (n+1) /2, with n = sym.shape[0]

    Parameters
    ==========
    sym: array
        shape (..., n, n)
    isometry: bool, optional, default to True
        used map is an isometry or not

    Returns
    =======
    vec: array
        shape (..., n * (n+1) /2)
    """
    p = sym.shape[-1]
    tril_mask = np.tril(np.ones(sym.shape[-2:])).astype(np.bool)
    sym_copy = copy.copy(sym)
    if isometry:
        off_diag_mask = (np.ones((p, p)) - np.eye(p)).astype(np.bool)
        sym_copy[..., off_diag_mask] *= np.sqrt(2)

    return sym_copy[..., tril_mask]


def vec_to_sym(vec, isometry=True):
    n = vec.size
    # solve p * (p + 1) / 2 = n subj. to p > 0
    # p ** 2 + p - 2n = 0 & p > 0
    # p = - 1 / 2 + sqrt( 1 + 8 * n) / 2
    p = (np.sqrt(8 * n + 1) - 1.) / 2
    try:
        np.testing.assert_almost_equal(p, int(p))
    except AssertionError:
        raise ValueError("Vector size unsuitable, can not transform vector to "
                         "symmetric matrix")

    p = int(p)
    tril_mask = np.tril(np.ones((p, p))).astype(np.bool)
    off_diag_mask = (np.ones((p, p)) - np.eye(p)).astype(np.bool)
    sym = np.zeros((p, p), dtype=np.float)
    sym[..., tril_mask] = vec
    sym.T[..., tril_mask] = vec
    if isometry:
        sym[..., off_diag_mask] /= np.sqrt(2)

    return sym


def cov_to_corr(cov):
    return cov * np.diag(cov) ** (-1. / 2) *\
        (np.diag(cov) ** (-1. / 2))[..., np.newaxis]


def prec_to_partial(prec):
    partial = -cov_to_corr(prec)
    np.fill_diagonal(partial, 1.)
    return partial


class CovEmbedding(BaseEstimator, TransformerMixin):
    """ Tranformer that returns the coefficients on a flat space to
    perform the analysis.
    """

    def __init__(self, cov_estimator=None, kind='tangent'):
        self.cov_estimator = cov_estimator
        self.kind = kind

    def fit(self, X, y=None):
        if self.cov_estimator is None:
            self.cov_estimator_ = EmpiricalCovariance(
                assume_centered=True)
        else:
            self.cov_estimator_ = clone(self.cov_estimator)

        if self.kind == 'tangent':
            covs = [self.cov_estimator_.fit(x).covariance_ for x in X]
            self.mean_cov_ = spd_mfd.frechet_mean(covs, max_iter=30, tol=1e-7)
            self.whitening_ = spd_mfd.inv_sqrtm(self.mean_cov_)
        return self

    def transform(self, X):
        """Apply transform to covariances

        Parameters
        ----------
        covs: list of array
            list of covariance matrices, shape (n_rois, n_rois)

        Returns
        -------
        list of array, transformed covariance matrices,
        shape (n_rois * (n_rois+1)/2,)
        """
        covs = [self.cov_estimator_.fit(x).covariance_ for x in X]
        covs = spd_mfd.my_stack(covs)
        if self.kind == 'tangent':
            covs = [spd_mfd.logm(self.whitening_.dot(c).dot(self.whitening_))
                    for c in covs]
        elif self.kind == 'precision':
            covs = [spd_mfd.inv(g) for g in covs]
        elif self.kind == 'partial correlation':
            covs = [prec_to_partial(spd_mfd.inv(g)) for g in covs]
        elif self.kind == 'correlation':
            covs = [cov_to_corr(g) for g in covs]
        else:
            raise ValueError("Unknown connectivity measure.")

        return np.array([sym_to_vec(c) for c in covs])
