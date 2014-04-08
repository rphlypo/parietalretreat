# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 10:31:13 2014

@author: sb238920
"""
from __future__ import print_function

import numpy as np

from second_level_new import Comparison, corr_to_Z


def results(fcs, tests, corrected=False): # TODO: add masking option
    cond_names = set(cond_name for cond_name in fcs.keys()[0])
    measure_names = [cond_name for measure_name in fcs.keys()[1]]
    identity = lambda x: x
    transforms = {'correlations':corr_to_Z,
                  'partial correlations':corr_to_Z,
                  'covariances':identity,
                  'precisions':identity,
                  'tangent plane':identity}
    results_ = {}                    
    for the_cond_names, the_covs in tests:
        # sanity check
        if not set(the_cond_names).issubset(cond_names):
            print('One or more conditions named', end=' ')
            print(*the_cond_names, sep=', ', end=' ')
            print('do not exit.')
            continue

        the_cov_names = the_covs.keys()
        the_cov_values = the_covs.values()
        test_name = the_cov_names[0]+' '+the_cond_names[
        0]+' vs '+the_cov_names[-1]+' '+the_cond_names[-1]
        for measure_name in self.measure_names: #set(zip(*self.fc_.keys())[1]):
            the_fcs = [self.fc_.get((the_cond_name,measure_name))
            for the_cond_name in the_cond_names]
            #the_fcs = [np.array(the_fc)[np.newaxis,...] for the_fc in the_fcs]
            comp = Comparison(the_cov_values, the_fcs,
                              transforms[measure_name], corrected)
            comp.stats()      
            results_[(test_name,measure_name)] = comp.stats_difference_
        print(test_name+' compared.')                
    return results_
        
if __name__ == "__main__":
    df, region_signals = load_data(root_dir="/media/Elements/volatile/new/salma",
                           data_set="ds107")
    df2 = get_region_signals(df, region_signals)
    groups = df2.groupby("condition")
    cond_names = []
    all_cond_signals = []
    for condition, group in groups:
        cond_names.append(condition)
        the_cond_signals=[]
        for ix_ in range(len(group)):
            the_cond_signals.append(group.iloc[ix_]["region_signals"])
        all_cond_signals.append(the_cond_signals)
        
    fcs = {}
    for cond_name, region_signals in zip(cond_names, all_cond_signals):
        print(cond_name)
        fc = analysis(region_signals, True, "covariances",
                      "precisions", "tangent plane")
        for measure_name, measure_values in fc.iteritems():                    
            fcs[(cond_name, measure_name)] = measure_values            
    if False:
        group_all = np.ones((1,n_subjects))
        group = {'All': group_all}
        words = 'Words'
        objects = 'Objects'          
        paired_tests = [([words, objects], group)]
        mc.results(paired_tests)