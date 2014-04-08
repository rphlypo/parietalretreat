# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 09:24:53 2014

@author: sb238920
TO improve by creating classes setup, preproc and link functions to them
"""
import glob
import os.path

import numpy as np

from compute_precision import idx_template  # to change to general module
import matlab_utilities as mat

def count_preproc(project_path):
    """Gives the number of subjects and runs in the preprocessing folder.
    The number of conditions for each subject is assumed the same.
    Parameters
    ==========
    project_path: u
        path of the conn project
        
    Returns
    =======
    n_subjects: int
        number of subjects
    n_runs: int
        number of conditions
    """
    preproc_path = os.path.join(project_path,'results/preprocessing')    
    paths =  os.path.join(preproc_path,'ROI_Subject*_Condition001.mat')
    n_subjects = len(glob.glob(paths))    
    paths =  os.path.join(preproc_path, 'ROI_Subject001_Condition*.mat') 
    n_runs = len(glob.glob(paths))
    return n_runs, n_subjects
    

def network(file_path, template_network):
    """Gives the list of lists of present roi names in file 
    'ROI_Subject*_Condition001.mat' in the preprocessing folder in comparison
    with the list of lists template_network
    
    Parameters
    ==========
    template_network: list of tuples (network name, rois)
        network name: str, network names
        rois: list of str, rois names
        
    Returns
    =======
    network: list of tuples (present network name, present rois)
        network name: str, element of template_network specifying present 
        networks
        values: list of str, subset of rois of template_network values for 
        each network specifying present rois 
    """
    
    if not os.path.isfile(file_path):
        ValueError('No such file \n')

    kwargs = {'names': 'cellstr'}     
    conn_rois = mat.import_matlab_data(file_path,**kwargs)['names']
    network = []
    for ntwk, rois in template_network:
        present_idx = idx_template(conn_rois,rois)
        if present_idx:
            network.append((ntwk, [conn_rois[idx] for idx in present_idx]))

    return network
    

def cond_name(file_path):
    """To vectorize    
    """                                
    kwargs = {'conditionname':'str'}     
    imp = mat.import_matlab_data(file_path,**kwargs)    
    return str(imp['conditionname'])

    
def extract_rois(file_path,rois):
    """
    Parameters
    ==========
    file_path: u
        file path 'ROI_Subject%03d_Condition%03d.mat'
        in folder results/preprocessing
    rois: list of str
        names of the rois to include
    
    Returns
    =======        
    signals: array, shape (n_samples,n_features)
    """              
    if not os.path.isfile(file_path):
        ValueError("no such file")
        
    kwargs = {'data': 'cell','names': 'cellstr'}   
    imp = mat.import_matlab_data(file_path,**kwargs)
    conn_rois = imp['names']
    conn_signals = imp['data']
    n_samples = len(conn_signals[0]) #conn_signals.shape[0]
    signals = np.empty((n_samples,0))        
    for roi in rois:
        if conn_rois.__contains__(roi):
            signals = np.concatenate(
            (signals,
             conn_signals[conn_rois.index(roi)][:,0][:,np.newaxis]),axis=1)  
         
    return signals
    

def covariates(conn_dir):
    conn_file = conn_dir + '.mat'
    kwargs = {'CONN_x': 'struct'}  
    contents = mat.import_matlab_data(conn_file,**kwargs) 
    CONN_x = contents['CONN_x']
    cov_names = CONN_x.Setup.l2covariates.names
    cov_values = CONN_x.Setup.l2covariates.values
    cov = {}
    for n, name in enumerate(cov_names):
        if name != ' ':
            cov[str(name)] = np.array([int(value[n]) for value in cov_values])
    return cov
 

def conditions(conn_dir):
    conn_file = conn_dir + '.mat'
    kwargs = {'CONN_x': 'struct'}  
    contents = mat.import_matlab_data(conn_file,**kwargs) 
    CONN_x = contents['CONN_x']
    conditions = [str(name) for name in CONN_x.Setup.conditions.names 
    if name != ' ']
    return conditions      
