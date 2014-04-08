# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 15:41:10 2014

@author: sb238920
Test exitence of conn_folder
test existence of conditions names
add a folder tests
"""
from __future__ import print_function

import numpy as np 

from my_conn import MyConn
from first_level import FC 

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

if __name__ == '__main__':
    AN = ['vIPS_big','pIPS_big','MT_big','FEF_big','RTPJ','RDLPFC'] # biyu's order
    DMN = ['AG_big','SFG_big','PCC','MPFC','FP'] # biyu's order
    final = 1    
    if final:
    #        WMN = ['IPL','MFG_peak1_peak2','LMFG_peak1','RMFG_peak2',
    #              'CPL_peak1_peak3','RCPL_peak1','RCPL_peak2','LCPL_peak3','LT']        
        #WMN = ['IPL','LMFG_peak1','CPL_peak1_peak3','LT']   
        WMN = ['IPL','LMFG_peak1',#'RMFG_peak2',
               'RCPL_peak1','LCPL_peak3','LT']  
    else:     
        WMN = ['RT','LT','PL.cluster002','LFG',
               'MFG.cluster002','LPC.cluster001',
               'SPL.cluster002'] # main peaks in SPM.mat 
            
    template_ntwk = [('WMN',WMN), ('AN',AN), ('DMN',DMN)]
    conn_folder = '/volatile/new/salma/subject1to40/conn_servier2_1to40sub_RS1-Nback2-Nback3-RS2_Pl-D_1_1_1'  
    mc = MyConn('from_conn', conn_folder)
    mc.setup()
    #fake_mc = MyConn(setup='test')
    standardize = True
    cond_names = ['ReSt1_placebo']
    all_conds_signals = mc.runs_['ReSt1_placebo']
    fcs = {}
    for cond_name, region_signals in zip(cond_names, all_conds_signals):
        print('ok')
        fc = analysis(region_signals, standardize, "covariances")
        for measure_name, measure_values in fc.iteritems:                    
            fcs[(cond_name, measure_name)] = measure_values            
    if False:    
        mc.analysis(template_ntwk, standardize, 'correlations', 'partial correlations', 'segregations')
                    #'variability')   
        mc.analysis_fig("/home/sb238920/slides/servierFinal/Images_tmp/", 'overwrite', 1,
                        ['Nbac3_Placebo'])            
        # between subjects covariates    
        n_subjects_A1 = 21
        n_subjects_A2 = 19
        n_subjects = 40
        group_all = np.ones((1,n_subjects))
        group_A1 = np.hstack(
        (np.ones((1,n_subjects_A1)),np.zeros((1,n_subjects_A2))))
        group_A2 = np.hstack(
        (np.zeros((1,n_subjects_A1)),np.ones((1,n_subjects_A2))))
        groups = {'All': group_all,
                  'A1': group_A1,
                  'A2': group_A2,}
        RS1_pl = 'ReSt1_Placebo'
        NB2_pl = 'Nbac2_Placebo'          
        paired_tests = [([RS1_pl, NB2_pl], ['All'])]
        indep_tests = [([RS1_pl], ['A1', 'A2'])]
        mc.results(paired_tests+indep_tests) #  add masking option
    #    mc.performances(perfs_file, test)
    #    overwrite = True
    #    mc.analysis_fig = (overwrite,'/home/sb238920/slides/servier2/Images')
        
