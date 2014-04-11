# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 14:19:55 2014

@author: sb238920
"""

import numpy as np
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg import logm

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.covariance import EmpiricalCovariance


def my_stack(arrays):
    return np.concatenate([a[np.newaxis] for a in arrays])

# Bypass scipy for faster eigh (and dangerous: Nan will kill it)
my_eigh, = get_lapack_funcs(('syevr', ), np.zeros(1))


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


def sym_to_vec(sym):
    """ Returns the lower triangular part of
    sqrt(2) (sym - Id) + (1-sqrt(2)) * diag(sym - Id),

    sqrt(2) * offdiag(sym) + np.diag(np.diag(sym)) - Id

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
    sym = np.empty((p, p), dtype=np.float)
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
            # self.mean_cov = spd_manifold.log_mean(covs)
            # Euclidean mean as an approximation to the geodesic
            covs = [self.cov_estimator_.fit(x).covariance_ for x in X]
            covs = my_stack(covs)
            mean_cov = np.mean(covs, axis=0)
            self.whitening_ = inv_sqrtm(mean_cov)
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
        covs = my_stack(covs)
        if self.kind == 'tangent':
            #p = covs.shape[-1]
            #id_ = np.identity(p)
            #covs = [self.whitening_.dot(c.dot(self.whitening_)) - id_
            #        for c in covs]      # Linearization of the exponential
            covs = [logm(self.whitening_.dot(c.dot(self.whitening_)))
                    for c in covs]
        elif self.kind == 'precision':
            covs = [inv(g) for g in covs]
        elif self.kind == 'partial correlation':
            covs = [np.fill_diagonal(-cov_to_corr(inv(g)),1.) for g in covs]
        elif self.kind == 'correlation':
            covs = [cov_to_corr(g) for g in covs]
        else:
            raise ValueError("Unknown connectivity measure.")
            
        return np.array([sym_to_vec(c) for c in covs])