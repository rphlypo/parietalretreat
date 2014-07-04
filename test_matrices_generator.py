import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less
from nose.tools import assert_not_equal
from scipy import linalg

import matrices_generator as my_gen


def test_random_diagonal_spd():
    """Testing random_diagonal_spd function"""
    d = my_gen.random_diagonal_spd(15)
    diag = np.diag(d)
    assert_array_almost_equal(d, np.diag(diag))
    assert_array_less(0.0, diag)


def test_random_spd():
    """Testing random_spd function"""
    spd = my_gen.random_spd(17)
    assert_array_almost_equal(spd, spd.T)
    assert(np.all(np.isreal(spd)))
    assert_array_less(0., np.linalg.eigvalsh(spd))


def test_random_non_singular():
    """Testing random_non_singular function"""
    non_sing = my_gen.random_non_singular(23)
    assert_not_equal(linalg.det(non_sing), 0.)