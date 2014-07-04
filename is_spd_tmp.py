# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ce script temporaire est sauvegardé ici :
/home/sb238920/.spyder2/.temp.py
"""

def is_spd(M, decimal=15):
    """Assert that input matrix is symmetric positive definite.

    M must be symmetric down to specified decimal places.
    The check is performed by checking that all eigenvalues are positive.

    Parameters
    ==========
    M: numpy.ndarray
        symmetric positive definite matrix.

    Returns
    =======
    answer: boolean
        True if matrix is symmetric positive definite, False otherwise.
    """
    if not np.allclose(M, M.T, atol=0.1 ** decimal):
        print("matrix not symmetric to {0} decimals".format(decimal))
        return False
    if np.all(np.iscomplex(M)):
        print("matrix has a non real value {0}".format(M[np.iscomplex(M)][0]))
    eigvalsh = np.linalg.eigvalsh(M)
    ispd = eigvalsh.min() > 0
    if not ispd:
        print("matrix has a negative eigenvalue: %.3f" % eigvalsh.min())
    return ispd

        try:
            assert_array_almost_equal(mat, mat.T)
            assert(np.all(np.isreal(mat)))
            assert_array_less(0.0, np.linalg.eigvalsh(mat))
        except AssertionError:
            raise ValueError("at least one matrix is not real spd")

