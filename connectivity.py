import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EmpiricalCovariance

import manifold as spd_mfd


def sym_to_vec(sym):
    """ Returns the lower triangular part of
    sqrt(2) sym + (1-sqrt(2)) * diag(sym),

    sqrt(2) * offdiag(sym) + np.diag(np.diag(sym))

    shape n (n+1) /2, with n = sym.shape[0]

    Parameters
    ==========
    sym: array
    """
    sym = np.sqrt(2) * sym
    # the sqrt(2) factor
    p = sym.shape[-1]
    sym.flat[::p + 1] = sym.flat[::p + 1] / np.sqrt(2)
    mask = np.tril(np.ones(sym.shape[-2:])).astype(np.bool)
    return sym[..., mask]


def vec_to_sym(vec):
    n = vec.size
    # solve p * (p + 1) / 2 = n subj. to p > 0
    # p ** 2 + p - 2n = 0 & p > 0
    # p = - 1 / 2 + sqrt( 1 + 8 * n) / 2
    p = int((np.sqrt(8 * n + 1) - 1.) / 2)
    mask = np.tril(np.ones((p, p))).astype(np.bool)
    sym = np.zeros((p, p), dtype=np.float)
    sym[..., mask] = vec
    sym = (sym + sym.T) / np.sqrt(2)  # divide by 2 multiply by sqrt(2)
    sym.flat[::p + 1] = sym.flat[::p + 1] / np.sqrt(2)
    return sym


def cov_to_corr(cov):
    return cov * np.diag(cov) ** (-1. / 2) *\
        (np.diag(cov) ** (-1. / 2))[..., np.newaxis]


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
            covs = [np.fill_diagonal(-cov_to_corr(spd_mfd.inv(g)), 1.) for g in
            covs]
        elif self.kind == 'correlation':
            covs = [cov_to_corr(g) for g in covs]
        else:
            raise ValueError("Unknown connectivity measure.")

        return np.array([sym_to_vec(c) for c in covs])