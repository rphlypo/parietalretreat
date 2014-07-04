# -*- coding: utf-8 -*-
"""
Created on Thu May 29 07:54:12 2014

@author: sb238920
"""
# Standard library imports

# Related third party imports
import numpy as np
from sklearn.utils import check_random_state

# Local application/library specific imports
import test_manifold


def make_sparse_spd(n_subjects=5, n_features=30, density=0.1, random_state=0):
    random_state = check_random_state(random_state)
    # Generate topology (upper triangular binary matrix, with zeros on the
    # diagonal)
    topology = np.empty((n_features, n_features))
    topology[:, :] = np.triu((
        random_state.randint(0, high=int(1. / density),
                         size=n_features * n_features)
    ).reshape(n_features, n_features) == 0, k=1)

    # Generate edges weights on topology
    precisions = []
    mask = topology > 0
    mask += mask.T
    mask += int(np.sum(mask > 0)) * np.eye(n_features)
    for _ in range(n_subjects):

        # See also sklearn.datasets.samples_generator.make_sparse_spd_matrix
        prec = test_manifold.random_spd(n_features)
        prec = mask * prec

        # Assert precision matrix is spd
        np.testing.assert_almost_equal(prec, prec.T)
        eigenvalues = np.linalg.eigvalsh(prec)
        if eigenvalues.min() < 0:
            raise ValueError("Failed generating a positive definite precision "
                             "matrix. Decreasing n_features can help solving "
                             "this problem.")
        precisions.append(prec)


if __name__ == "__main__":
    density = 0.1
    n_features = 30
    make_sparse_spd(n_subjects=5, n_features=n_features, density=density,
                    random_state=0)