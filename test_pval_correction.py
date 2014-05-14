# Standard library imports
import random

# Related third party imports
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises_regexp, assert_true

# Local application/library specific imports
import pval_correction as my_crt


def test_correct():
    """Testing function correct"""
    # Random input 1-D array p of random size and with 2 nan enteries
    size = random.randint(1, 5)
    p = np.random.rand(size, )
    idx = random.randint(0, int(size / 3))
    p[idx] = np.nan
    p[2 * idx] = np.nan
    p_isnan = np.zeros(p.shape, dtype=bool)
    p_isnan[idx] = True
    p_isnan[2 * idx] = True
    p_isnotnan = np.logical_not(p_isnan)

    # Test output for different correction values
    q_uncorrected = my_crt.correct(p)
    q_bonferroni = my_crt.correct(p, correction="bonferroni")
    q_fdr = my_crt.correct(p, correction="fdr")
    assert_array_almost_equal(q_uncorrected, p)
    for q in [q_uncorrected, q_bonferroni, q_fdr]:
        assert_array_equal(p_isnan, np.isnan(q))
        assert_true(np.all(p[p_isnotnan] <= q[p_isnotnan]))

    # Check error is raised for unkown value of correction
    with assert_raises_regexp(ValueError, "Unknown correction."):
        my_crt.correct(p, correction="blabla")


def test_fdr():
    """Testing function fdr"""