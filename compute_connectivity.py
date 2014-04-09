# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:31:13 2014

@author: sb238920
"""
from __future__ import print_function

import numpy as np
import copy
from scipy import linalg

from classify_covs import load_data, get_region_signals
from covariance import CovEmbedding
from sklearn.base import BaseEstimator

import covariance


def cov_to_corr(cov):
    """
    Computes correlation from covariance

    Parameters
    ==========
    cov: np.array
        covariance matrix

    Returns
    =======
    corr: np.array
        correlation matrix
    """
    d = np.sqrt(np.diag(cov))
    corr = cov / d
    corr = corr / d[:, np.newaxis]
    return corr


def prec_to_partial_corr(prec):
    """
    Computes partial correlations from precision matrix.

    Parameters
    ==========
    prec: np.array
        precision matrix

    Returns
    =======
    partial_corr: np.array
        partial correlation matrix. Formulae is partial_corr[i,i] =1 and
        partial_corr[i,j] = - prec[i,j] / sqrt(prec[i,i] * prec[j,j])
        for i!= j
    """
    d = np.sqrt(np.diag(prec))
    partial_corr = prec / d
    partial_corr = partial_corr / d[:, np.newaxis]
    partial_corr *= -1
    np.fill_diagonal(partial_corr, 1)
    return partial_corr


class FC(BaseEstimator, TransformerMixin):
    """Functional connectivity class

    Attributes
    ----------
    `signals`: array
        ROIs time series, shpae (n_samples, n_features)
    `standardize`: bool, default to False
        standardize time series
    """
    def __init__(self, standardize=False, estimator=None):
        self.standardize = standardize
        self.base_estimator = base_estimator


    def fit(self, X):
        X = X - X.mean(axis=0)
        self.emp_cov_ = EmpiricalCovariance(X, assume_centered=True)


    def transform(self, X, y):
        if estimator is None:
            estimator = "tangent"
        if estimator == "tangent":
        elif estimator == "covariance":
        elif estimator == "precision":
        elif estimator == "correlation":
        elif estimator == "partialcorr":



    def compute_cov(self):
        """Compute empirical covariances
        Attributes
        ----------
        cov_: array
            covariance matrix, shape (n_features, n_features)
        """
        subject = copy.copy(self.signals)
        subject -= subject.mean(axis=0)
        if self.standardize:
            subject = subject / subject.std(axis=0)  # copy on purpose

        n_samples = subject.shape[0]
        self.cov_ = np.dot(subject.T, subject) / n_samples
        return self

    def compute_corr(self):
        """Compute empirical correlation matrix
        Attributes
        ----------
        corr_: array
            correlation matrix, shape (n_features, n_features)
        """
        self.corr_ = cov_to_corr(self.cov_)
        return self

    def compute_prec(self):
        """Compute empirical precision matrix
        Attributes
        ----------
        prec_: array
            precision matrix, shape (n_features, n_features)
        """
        cond_number = np.linalg.cond(self.cov_)
        if cond_number > 100:  # 1/sys.float_info.epsilon:
            print('Bad conditioning! ' +
                  'condition number is {}'.format(cond_number))
        self.prec_ = linalg.inv(self.cov_)

        return self

    def compute_partial(self):
        """Compute empirical partial correlation matrix
        Attributes
        ----------
        partial_corr_: array
            partial correlation matrix, shape (n_features, n_features)
        """
        self.partial_corr_ = prec_to_partial_corr(self.prec_)
        return self

    def compute_tangent(self):
        """Compute empirical partial correlation matrix
        Attributes
        ----------
        tangent_: array
            projection of the covariance matrix on the tangent plane,
            shape (n_features, n_features)
        """
        ce = CovEmbedding()
        ce.fit([self.cov_])
        self.tangent_ = ce.transform([self.cov_])[0]
        return self

    def compute(self, *args):
        """Computes the specified connectivity measures
        Parameters
        ----------
        *args: list of str
            measures names
        Returns
        -------
        self.conn_: dict
            keys: str, measures names
            values: array, measures values
        """
        computs = {0: self.compute_cov(), 1: self.compute_corr(),
                   2: self.compute_prec(), 3: self.compute_partial(),
                   5: self.compute_tangent()}
        measures_steps = {'correlations': [0, 1],
                          'partial correlations': [0, 2, 3],
                          'covariances': [0],
                          'precisions': [0, 2],
                          'tangent plane': [0, 5]}
        steps = [step for name in args for step in measures_steps[name]]
        steps = set(steps)
        for n_step in steps:
            computs[n_step]
        output = {'correlations': self.corr_,
                  'partial correlations': self.partial_corr_,
                  'covariances': self.cov_,
                  'precisions': self.prec_,
                  'tangent plane': self.tangent_}
        self.conn = {}
        for measure_name in args:
            self.conn[measure_name] = output[measure_name]
        return self


def analysis(region_signals, standardize=False, *args):
    """ Computes for given signals the connectivity matrices for specified
    measures

    Parameters
    ----------
    region_signals: array or list of region_signals
        regions time series, shape of each array n_samples, n_regions
    standardize: bool (optional, default to False)
        standardize roi signals or not
    *args: optional str, default to "covariances"
        names of the connectivity measures.

    Returns
    -------
    fc_: dict
        keys: str, names of connectivity measures
        values: array, shape n_subjects, n_regions, n_regions
                associated connectivity values,
    """
    if type(region_signals) == 'numpy.ndarray':
        region_signals = [region_signals]

    n_subjects = len(region_signals)
    print('{} subjects'.format(n_subjects))
    fc_ = {}
    for n_subject, subject in enumerate(region_signals):
        if n_subject == 0:
            fcs = []

        myFC = FC(subject, standardize)
        myFC.compute(*args)
        for n_measure, measure_name in enumerate(args):
            if n_subject == 0:
                print(measure_name)
                n_features = myFC.conn[measure_name].shape[0]
                print("{}, {}".format(n_features, n_subjects))
                fcs.append(np.empty((n_subjects, n_features, n_features)))

            fcs[n_measure][n_subject] = myFC.conn[measure_name]
    for n_measure, measure_name in enumerate(args):
        fc_[measure_name] = fcs[n_measure]
    print('\ncomputed measures: ', end='')
    print(*args, sep=', ')
    return fc_


if __name__ == "__main__":

    # Load conditions names and ROIs time series
    df, region_signals = load_data(
        root_dir="/home",  # /media/Elements/volatile/new/salma",
        data_set="ds107")
    df2 = get_region_signals(df, region_signals)
    groups = df2.groupby("condition")
    cond_names = []
    all_cond_signals = []
    for condition, group in groups:
        cond_names.append(condition)
        the_cond_signals = []
        for ix_ in range(len(group)):
            the_cond_signals.append(group.iloc[ix_]["region_signals"])
        all_cond_signals.append(the_cond_signals)

    # Compute connectivity matrices for each condition and each subject
    fcs = {}
    for cond_name, region_signals in zip(cond_names, all_cond_signals):
        print(cond_name)
        fc = analysis(region_signals, True, "covariances",
                      "precisions", "tangent plane")
        for measure_name, measure_values in fc.iteritems():
            fcs[(cond_name, measure_name)] = measure_values
