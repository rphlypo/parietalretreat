# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 11:59:01 2014

@author: sb238920
"""
import numpy as np


def compute_confounds(movconf_file):
    """Computes mouvement and linear trend confounds
    
    Parameters
    ==========
    dataset: str
        dataset, either "hcp", "henson2010faces", or openfmri datasets "
        ds*"
    mvt_dir: path 
        path for the directory containing mouvement parameters    
        
    Returns
    =======
    confounds_data: array
        mouvement parameters, derivatives and squares
    confounds_labels: list of str
        labels of the mouvement confounds
    """
    confounds_data = np.loadtxt(movconf_file, dtype=float)
    confounds_data = confounds_data[:6]  
    confounds_data_dt = confounds_data
    conv = np.convolve(confounds_data,np.array([1.,0.,-1.]/2),
                                    'valid')
    confounds_data_dt[1:-1] = conv
    pass

    confounds_data = np.concatenate(
        (confounds_data, confounds_data_dt), axis=1)
    confounds_data = np.concatenate(
        [confounds_data, confounds_data**2], axis=1)
    confounds_labels = ["trans_x", "trans_y",
                        "trans_z", "rot_x", "rot_y", "rot_z"]
    confounds_labels_dt = ["d_" + lbl for lbl in confounds_labels]
    confounds_labels_sq = ["sq_" + lbl for lbl in confounds_labels]
    confounds_labels.extend(confounds_labels_dt)
    confounds_labels.extend(confounds_labels_sq)
    return confounds_data, confounds_labels
