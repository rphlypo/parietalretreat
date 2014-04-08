# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:10:05 2014

@author: sb238920
Todo: variablity, semi-partial correlation
"""
import copy
import numpy as np
import matlab_utilities as mat
from scipy import linalg
from math import log

from matrix import untri
from covariance import CovEmbedding


def cov_to_corr(cov):
    """
    Computes correlation from covariance
    """    
    d = np.sqrt(np.diag(cov))
    corr = cov/d
    corr = corr/d[:, np.newaxis]
    return corr


def integ_to_seg(integ):
    """
    Computes segregation measures between all pairs of 
    networks for a given integration matrices.
    
    Parameters
    ==========
    
    integ: numpy ndarray
        ndim 2, shape for each dimension equal to networks number
        integ[i,i] is integration for network i and integ[i,j] is integration for network i union network j
        
    Returns
    =======
    
    segreg: numpy ndarray
        ndim 2, shape for each dimension equal to networks number
        segreg[i,j] is  segregation for network i union network j  
        segreg[i,i] is integration for network i
    """
    d = integ[np.diag_indices_from(integ)]
    segreg = integ - d -d[:,None]   # segregation[i,j] = integration[i,j] - integration[i,i] - integration[j,j]       
    segreg *= -1                    # segregation[i,i] = integration[i,i]   
    return segreg     
    
    
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
        partial correlation matrix.     Formulae is partial_corr[i,i] =1 and
        partial_corr[i,j] = - prec[i,j] / sqrt(prec[i,i] * prec[j,j]) for i!= j       
    """    
    d = np.sqrt(np.diag(prec))
    partial_corr = prec/d
    partial_corr = partial_corr/d[:, np.newaxis]
    partial_corr *= -1
    np.fill_diagonal(partial_corr,1)
    return partial_corr          
    

def prec_to_integ(prec,ntwksDim):
    """
    Computes integration measures for each network and between all pairs of 
    networks for a given precision matrix and specified dimensions of each 
    network.  
    
    Parameters
    ==========    
    prec: numpy array
        precision matrix
    
    ntwksDim: numpy array
        dimensions of each network, ordered as in the precision matrix
        
    Returns
    =======    
    integ: numpy ndarray
        ndim 2, shape for each dimension equal to networks number
        integ[i,i] is integration for network i and integ[i,j] is integration for network i union network j     
    """
    n_ntwks = len(ntwksDim) # number of present networks
    slices = ntwksDim.cumsum()
    slices = np.concatenate(([int(0)],slices),axis=0)    
    # diag_indexes: list of length n_ntwks, each element of it is the list of indexes in 
    # the precision  matrix of ROIs for the corresponding network
    diag_indexes = [range(slices[nt],slices[nt+1]) for nt in range(n_ntwks)] 
    block_prec = np.empty((n_ntwks,n_ntwks) ,dtype = 'object')  # blocks of precision matrix, block_prec[i,i] is precision for network i and block[i,j] is precision for network i union network j        
    for i in range(n_ntwks):
        block_prec[i,i] = mat.select(
        prec, diag_indexes[i])
        for j in range(i):
            block_prec[i,j] = mat.select(
            prec,diag_indexes[i]+diag_indexes[j])
            block_prec[j,i] = mat.select(
            prec,diag_indexes[j]+diag_indexes[i])       
    compute_integ = np.vectorize(lambda x: 0.5*log(linalg.det(x)))
    integ = compute_integ(block_prec) # integration matrices, integ[i,i] is integration for network i and integ[i,j] is integration for network i union network j    
    return integ  

    
class FC(object):
    """Functional connectivity class
    
    Attributes
    ----------
    signals: array
        ROIs time series
    """        
    def __init__(self, signals, standardize=True, ntwk_dims=None):
        self.signals = signals
        self.standardize = standardize
        self.ntwk_dims = ntwk_dims
        
    def set_params(self, **kwargs):
        for param, val in kwargs.items():
            self.setattr(param,val)

    def get_params(self, deep=True):
        return {"signals"     : self.signals,
                "standardize" : self.standardize,
                "ntwk_dims"   : self.ntwk_dims}
        
    def compute_cov(self):
        subject = copy.copy(self.signals)
        subject -= subject.mean(axis=0)     
        if self.standardize:
            subject = subject/subject.std(axis=0) # copy on purpose
        
        n_samples = subject.shape[0]
        self.cov = np.dot(subject.T, subject) / n_samples # covariance matrix  
        return self

    def compute_corr(self):
        self.corr = cov_to_corr(self.cov) # partial correlation matrix            
        return self
        
    def compute_prec(self):            
        ce = CovEmbedding('precisions')
        self.prec = ce.fit_transform(self.cov)
        if False:
            cond_number = np.linalg.cond(self.cov) 
            if cond_number > 100:#1/sys.float_info.epsilon:
                print('Bad conditioning! condition number is {}'.format(cond_number))
            self.prec = linalg.inv(self.cov) # precision matrix  
            
        return self    

    def compute_partial(self):
        self.partial_corr = prec_to_partial_corr(self.prec) # partial correlation matrix            
        return self
        
    def compute_seg(self):
       integ = prec_to_integ(self.prec,np.array(self.ntwk_dims)) # integration matrices     
       self.seg = integ_to_seg(integ)  # segregation matrices 
       return self

    def compute_tangent(self):
        ce = CovEmbedding()
        self.tangent = ce.fit_transform(self.cov)
        return self

    def compute(self,*args):
        """Computes the specified connectivity measures
        Parameters
        ----------
        *args: list of str
            measures names
        Returns
        -------
        self.conn_: dict
            keys: str, measures names
            values: arrys, measures values
        """
        computs = {0:self.compute_cov(), 1:self.compute_corr(),
                   2:self.compute_prec(), 3:self.compute_partial(), 
                   4:self.compute_seg(), 5:self.compute_tangent()}
        measures_steps = {'correlations':[0,1], 'partial correlations':[0,2,3],
                          'segregations':[0,2,4],'covariances':[0],
                          'precisions':[0,2], 'tangent space':[0,5]}        
        steps = [step for name in args for step in measures_steps[name]]  
        steps = set(steps)         
        for n_step in steps:
            computs[n_step]
        output = {'correlations':self.corr,
                  'partial correlations':self.partial_corr,
                  'segregations':self.seg,
                  'covariances':self.cov,
                  'precisions':self.prec,
                  'tangent space':self.tangent} 
        self.conn = {}       
        for measure_name in args:
            self.conn[measure_name] = output[measure_name]
        return self
            