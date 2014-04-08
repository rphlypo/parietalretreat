# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 17:25:21 2014

@author: sb238920
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:19:13 2014

@author: sb238920
To improve with inheritance
To add Shapiro-Wilk and equal variance testing
To remove the axes attributes
"""
import copy
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from compute_precision import plot_matrix

#def fdr(p_vals)
class Comparison(object):
    """ Between and within groups comparisons
    Parameters
    ----------
    subj_reg: list of array,
        between subjects regressors
    conditions: list of array,
        run(s) to compare
            
    """
    def __init__(self, subj_reg, conditions, transform=lambda x: x,
                 corrected=False, subj_axis=0, cond_axis=1):
        self.subj_reg = subj_reg  # between subjects covariates
        self.conditions = conditions
        self.transform = transform
        self.corrected = corrected
        self.subj_axis = subj_axis
        self.cond_axis = cond_axis
        
    def set_params(self, **kwargs):
        for param, val in kwargs.items():
            self.setattr(param,val)

    def get_params(self, deep=True):
        return {"subj_reg"     : self.subj_reg,
                "conditions"   : self.conditions,
                "transform"    : self.transform,
                "corrected"    : self.corrected,
                "subj_axis"    : self.subj_axis,
                "cond_axis"    : self.cond_axis }   
               
    def stats(self):
        self.set_baseline()
        self.set_follow_up()
        self.stats_baseline_ = self.stats_baseline()        
        self.stats_follow_up_ = self.stats_follow_up()        
        self.stats_difference_ = self.stats_difference()
        return self
        
    def threshold(self, th):
        self.signif_baseline_ = self.signif_baseline(th)
        self.signif_follow_up_ = self.signif_follow_up(th)
        self.signif_difference_ = self.signif_difference(th)
        return self
               
    def set_baseline(self):
        """    
        subjects_axis: int, optional
            number of the subjects axis, default to 0
        cond_axis: int, optional
            number of the conditions axis, default to 1
        """
#        axis = range(self.conditions.ndim)
#        axis.remove(self.subj_axis)
#        axis.remove(self.cond_axis)
 #       data = np.transpose(self.conditions,
 #                         [self.subj_axis, self.cond_axis]+axis)
        self._baseline = self.conditions[0][self.subj_reg[0] == 1,...]   
        return self                      
               
    def set_follow_up(self):           
#        axis = range(self.conditions.ndim)
#        axis.remove(self.subj_axis)
#        axis.remove(self.cond_axis)
#        data = np.transpose(self.conditions,
#                          [self.subj_axis,self.cond_axis]+axis)        
        self._follow_up = self.conditions[-1][self.subj_reg[-1] == 1,...]   
        return self
                      
    def stats_baseline(self):
        t, pval = stats.ttest_1samp(self.transform(self._baseline), 0.0, 
                                    self.subj_axis)
        if self.corrected:
            pval = fdr(pval)
        return t, pval

    def signif_baseline(self, th=0.05):
        t, pval = self.stats_baseline(self.subj_axis)
        return get_signif_effect(self._baseline.mean(self.subj_axis),pval,th)        
        
    def stats_follow_up(self, axis=0):
        t, pval = stats.ttest_1samp(self.transform(self._follow_up), 0.0,
                                    axis)        
        if self.corrected:
            pval = fdr(pval)
        return t, pval
      
    def signif_follow_up(self, th=0.05):
        t, pval = self.stats_follow_up(self.subj_axis)
        return get_signif_effect(self._follow_up.mean(self.subj_axis),pval,th)  
        
    def stats_difference(self):
        if len(self.subj_reg) > 1:
            ttest = stats.ttest_ind  # 2 independent samples, equal variance
        else:
            ttest = stats.ttest_rel 
        t, pval = ttest(self.transform(self._baseline), 
                        self.transform(self._follow_up),self.subj_axis) 
        if self.corrected:
            pval = fdr(pval)
        return t, pval 

    def signif_difference(self, th=0.05):        
        t, pval1 = self.stats_baseline(self.subj_axis)
        t, pval2 = self.stats_follow_up(self.subj_axis)
        t, pval = self.stats_difference(self.subj_axis)     
        pval_all = np.maximum(np.minimum(pval1,pval2),pval)
        effect = self._follow_up.mean(
        self.subj_axis)-self._baseline.mean(self.subj_axis)
        return get_signif_effect(effect,pval_all,th)  

    def plot(self, th=0.05, ticks=[], tickLabels=[],
             titles=["follow_up","baseline","difference"],
             ylabels=["","",""], abs_maxs=[None,None,None],
             abs_mins=[None,None,None], symmetric_cbs=[True,True,True]):
        fig = plt.figure(figsize=(10, 7))
        plt.subplots_adjust(hspace=0.8,wspace=0.5)      
        signifs = [self.signif_follow_up(th),
                   self.signif_baseline(th),
                   self.signif_difference(th)]
        matr = zip(titles,signifs)
        for i, (name, this_matr) in enumerate(matr):
            plt.subplot(1, 3, i+1)
            plot_matrix(this_matr, plt.gca(), ticks, tickLabels, name, 
                        ylabels[i], abs_maxs[i], abs_mins[i],
                        symmetric_cbs[i])
        return fig 
        
        
def get_signif_effect(effect, pval, th=0.05):
    """ Returns mean of an array sample mapped on the thresholded 
    significance matrix.

    Paramters
    =========
    sample: numpy array of floats
        effect sizes
    pval: numpy array of floats
        p-values, same dimension as sample
    th: float, optional
        significance level, default to 0.05
    Returns
    =======
    signif_effect: numpy array of float
        significant effects       
    """
    signif_effect = copy.copy(effect) # avoid side effects
    signif_effect[np.where(pval > th)] = 0
    return signif_effect
    
    
def fdr(p):
    """ FDR correction for multiple comparisons.
    
    Computes fdr corrected p-values from an array o of multiple-test false 
    positive levels (uncorrected p-values) a set after removing nan values, 
    following Benjamin & Hockenberg procedure.
    
    Parameters
    ==========
    p: np.array
        uncorrected pvals
    
    Returns
    =======
    pFDR: np.array
        corrected pvals
    """
    if p.ndim == 1:
        N1 = p.shape[0]
        q = np.nan+np.ones(p.shape)
        idx = p.argsort()
        sp = p[idx]
        N1 = np.sum(np.logical_not( np.isnan(p)))
        if N1 > 0:
            qt = np.minimum(1,N1*sp[0:N1]/(np.arange(N1)+1))
            min1 = np.inf
            for n in range(N1-1,-1,-1):
                min1 = min(min1,qt[n])
                q[idx[n]] = min1
    else:        
        q = np.array([fdr(p[j]) for j in range(p.shape[1])])
    return q
    
    
class ConnsPerfs(object):
    """

    """
    def __init__(self, names, conns, perfs, corrected = False):
        self.names = names  # seed names
        self.conns = conns  # connectivity values, shape (n_pairs, n_subjects)
        self.perfs = perfs  # performances across subjects
        self.corrected = corrected # FDR correction or not
        
    def set_params(self, **kwargs):
        for param, val in kwargs.items():
            self.setattr(param,val)

    def get_params(self, deep=True):
        return {"names"         : self.names,
                "conns"         : self.conns,
                "perfs"         : self.perfs,
                "corrected"    : self.corrected}   
                
    def corrs(self):
        n_conns = self.conns.shape[0]
        ps = np.empty((n_conns,))
        rs = np.empty((n_conns,))
        for n_conn in range(n_conns):
            r, p = stats.stats.pearsonr(self.conns[n_conn], self.perfs)
            ps[n_conn] = p
            rs[n_conn] = r
        if self.corrected:
            ps = fdr(ps)
        return rs, ps



def corr_to_Z(corr):
    """
    Gives the Z-Fisher transformed correlation matrix. Correlations 1 and -1 
    are transformed to nan.

    Parameters
    ==========
    corr: np.array
        correlation matrix    
    
    Returns
    =======
    Z: np.array
        Z-Fisher transformed correlation matrix    
    """
    eps = sys.float_info.epsilon # 1/1e9
    Z = copy.copy(corr)          # to avoid side effects
    corr_is_one = 1.0 - abs(corr) < eps
    Z[corr_is_one] = np.nan
    Z[np.logical_not(corr_is_one)] =  np.arctanh(corr[np.logical_not(corr_is_one)]) #0.5*np.log((1+corr[1.0 - corr >= eps])/(1-corr[1.0 - corr >= eps]))    
    return Z


def q_q_plot(x,fig_path):  
    """
    saves Q-Q plot of x to a specified file path.
    
    Parameters
    ==========
    x: array
        sample distribution
    fig_path: str
        file path for saving the figure
    """
   