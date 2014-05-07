import nose
import warnings
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_less, \
    assert_array_equal
from nose.tools import assert_not_equal, assert_tuple_equal, assert_equal, \
    assert_is_instance, assert_almost_equal
from scipy import linalg

import connectivity as my_con


def test_sym_to_vec():
    """Testing sym_to_vec function"""
    m = np.random.rand(15, 15)
    sym = m + m.T
    vec = my_con.sym_to_vec(sym)
    assert_array_almost_equal(my_con.vec_to_sym(vec), sym)

    
if __name__ == "__main__":
    nose.run()