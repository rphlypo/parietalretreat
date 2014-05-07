import random
import nose
import numpy as np
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises, assert_equal
import connectivity as my_con


def test_sym_to_vec():
    """Testing sym_to_vec function"""
    shape = random.randint(1, 50)
    m = np.random.rand(shape, shape)
    sym = m + m.T
    vec = my_con.sym_to_vec(sym)
    assert_array_almost_equal(my_con.vec_to_sym(vec), sym)


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


if __name__ == "__main__":
    nose.run()