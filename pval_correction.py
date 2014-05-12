# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 11:50:42 2014

@author: sb238920
"""
import numpy as np
    

def correct(p, correction = None):
    """Correction for multiple comparisons.  
    
    Parameters
    ----------
    p: array
        uncorrected p-values. If number of array dimensions is more than 1,
        correction is done along the first axis.
        
    correction: str, optional.
        correction method, either "fdr" or "benferroni". Default to None for
        uncorrecting.
        
    Returns
    -------
    q : array
        corrected p-values, same shape as p.
    """
    if correction is None:
        q = p 
    elif correction == "bonferroni":
        q = bonferroni(p)
    elif correction == "fdr":
        q = fdr(p)
    else: 
        raise ValueError("Unknown correction.")
        
    return q


def bonferroni(p):
    """ Bonferroni correction for multiple comparisons.

    Returns an array of estimated false discovery rates (set-level q-values) 
    from an array of multiple-test false positive levels (uncorrected 
    p-values), disregarding nan values.     

    Parameters
    ----------
    p: array
        uncorrected p-values.
    
    Returns
    -------
    q: array
        Bonferroni corrected p-values, same shape as p.
    """
    if p.ndim == 0:
        q = p
    elif p.ndim == 1:
        N1 = p.shape[0]
        q = np.nan + np.ones(p.shape)
        N1 = np.sum(np.logical_not(np.isnan(p)))
        if N1 > 0:
            q = N1 * p 
    else:        
        q = np.array([bonferroni(p[j]) for j in range(p.shape[-1])])

    return q


def fdr(p):
    """ FDR correction for multiple comparisons.
    
    Returns an array of estimated false discovery rates (set-level q-values) 
    from an array of multiple-test false positive levels (uncorrected 
    p-values), disregarding nan values. Following Benjamini-Hochberg 
    procedure.    
    
    Parameters
    ----------
    p: array
        uncorrected p-values.
    
    Returns
    -------
    q: array
        FDR-corrected p-values, same shape as p.
    """
    if p.ndim == 0:
        q = p        
    elif p.ndim == 1:
        N1 = p.shape[0]
        q = np.inf + np.ones(p.shape)
        idx = p.argsort()
        sorted_p = p[idx]
        N1 = np.sum(np.logical_not(np.isnan(p)))
        if N1 > 0:
            qt = np.minimum(1, N1 * sorted_p[:N1] / (np.arange(N1) + 1))
            min1 = np.inf
            for n in range(N1 - 1, -1, -1):
                min1 = min(min1, qt[n])
                q[idx[n]] = min1
    else:
        q = np.array([fdr(p[j]) for j in range(p.shape[1])])

    return q
