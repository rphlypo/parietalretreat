# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:31:13 2014

@author: sb238920
"""
from __future__ import print_function

import numpy as np

from first_level import FC 
from second_level_new import Comparison, corr_to_Z

def analysis(region_signals, standardize = False, *args):
    """ Computes for given signals the connectivity matrices measured with 
    the specified kinds

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
    measure_names = args
    fc_ = {}
    for subject, n_subject in enumerate(region_signals):
        ntwk_dims = None                
        if n_subject == 0:         
            fcs = []

        myFC = FC(subject, standardize, ntwk_dims)            
        myFC.compute(*args)
        for n_measure, measure_name in enumerate(args):
            if n_subject == 0:
                n_features = myFC.conn[measure_name].shape[0]
                fcs.append(np.empty((n_subjects,n_features,n_features)))
            
            fcs[n_measure][n_subject] = myFC.conn[measure_name]
    for n_measure, measure_name in enumerate(args):
        fc_[measure_name] = fcs[n_measure]
    print('\ncomputed measures: ',end='')
    print(*args,sep=', ')
    return fc_     


def results(all_cond_names, tests,corrected=False): # todo: add masking option
    # reading all conditions names
    cond_names = all_cond_names
    identity = lambda x: x
    transforms = {'correlations':corr_to_Z,
                  'partial correlations':corr_to_Z,
                  'semi-partial correlations':corr_to_Z,
                  'segregations':identity}
    results_ = {}                    
    for the_cond_names, the_cov_names in tests:
        if not set(the_cond_names).issubset(cond_names):
            print('One or more conditions named', end=' ')
            print(*the_cond_names,sep=', ', end=' ')
            print('do not exit.')
            continue
        
        the_covs = [covs[the_cov_name] 
        for the_cov_name in the_cov_names]
        test_name = the_cov_names[0]+' '+the_cond_names[
        0]+' vs '+the_cov_names[-1]+' '+the_cond_names[-1]
        for measure_name in self.measure_names: #set(zip(*self.fc_.keys())[1]):
            the_fcs = [self.fc_.get((the_cond_name,measure_name))
            for the_cond_name in the_cond_names]
            #the_fcs = [np.array(the_fc)[np.newaxis,...] for the_fc in the_fcs]
            comp = Comparison(the_covs, the_fcs,
                              transforms[measure_name], corrected)
            comp.stats()      
            results_[(test_name,measure_name)] = comp.stats_difference_
        print(test_name+' compared.')                
    return results_
        
if __name__ == "__main__":
    if False:
        fcs = {}
        for cond_name, region_signals in zip(cond_names, all_conds_signals):
            fc = analysis(region_signals, standardize = False, "covariances", 
                          "precisions", "tangent plane")
            for measure_name, measure_values in fc.iteritems:                    
                fcs[(cond_name, measure_name)] = measure_values            
    if False:
        group_all = np.ones((1,n_subjects))
        group = {'All': group_all}
        words = 'Words'
        objects = 'Objects'          
        paired_tests = [([words, objects], group)]
        mc.results(paired_tests)