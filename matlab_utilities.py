# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 14:30:47 2013

@author: sb238920
"""
from scipy import io
import numpy as np

def import_matlab_cell(file_path,**kwargs):
    """
     Reads the ROI names and the ROI signal for all subjects 
    from a specified CONN project and returns them respectively as
    a list and a list of numpy.ndarray.
    
    Parameters
    ==========    
    file_path: u
        Path of the Matlab file.
        
    var_name: str
        Name of the variable to extract.
        
    Returns
    =======
    presentNtwk: list of lists
        presentNtwk has the same length as ntwks
        presentNtwk[nt] is a list of unicode specifying the names of the ROIs
        included in the analysis
        
    subjects : list of numpy.ndarray, shape for each (1,nROIs)
        len(subjects) is the total number of subjects
        subjects[n] is the signals for subject n. 
        subjects[n][0,r] is the signal for subject n ROI r and has shape 
        (nsamples,nacp) where nacp is the number of ACP components for ROI r
   
    """
    mat = io.loadmat(file_path,squeeze_me=False)
    values = {}
    for var_name, var_type in kwargs.iteritems():
        var = mat[var_name] #numpy.ndarray
        if var_type == 'cellstr':
            var_out = var.transpose()
            length = len(var_out)      
            var_out = [var_out[string][0][0][:] for string in range(length)]
        else:           
            var_out = var[0]
        values[var_name] = var_out
        
    return values
    
def import_matlab_data(file_path,**kwargs):
    """
     Reads from a matlab file specified variables by their names and returns 
     them as numpy.ndarrays of shape 1 each. 
     If a variable type is specified as a cell, the returned variable is 
     a numpy.ndarray of shape as the size of the cell.  
     If a variable type is specified as a cellstr, the returned variable is 
     a list of length as the size of the cell.  
     
    Parameters
    ==========    
    file_path: u
        Path of the Matlab file.
        
    **kwargs: dict
        Keys are the names of the variables in the .mat file.
        Values are types of each variable. Recognized types for a special 
        treatment are 'cell' and 'cellstr'.
        Example: kwargs = {'data': 'cell','names': 'cellstr'} 
        
        
    Returns
    =======
    values: dict
        Keys are the names of the variables in the .mat file.
        Values are the corresponding python variables. They can be of type 
        numpy.ndarrays or lists.
   
    """
    mat = io.loadmat(file_path,squeeze_me=False)
    values = {}
    for var_name, var_type in kwargs.iteritems():
        var = mat[var_name] #numpy.ndarray
        if var_type == 'cellstr':
            var_out = var.transpose()
            length = len(var_out)      
            var_out = [var_out[string][0][0][:] for string in range(length)]
        elif var_type == 'cell':           
            var_out = var[0]
        elif var_type == 'str':
            var_out = var.transpose()
            var_out = [var_out[string][:] for string in range(len(var_out))][0]            
        elif var_type == 'struct':
            mat_tmp = io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
            var = mat_tmp[var_name]
            var_out = var
        else:
            var_out = var
            
        values[var_name] = var_out
        
    return values    
    
def select(array,indexes,indexes2=[]):
    """
    Masks a given np array with a list of indexes 

    Parameters
    ==========    

    array: numpy array
    
    indexes: list of int
        gives the indexes that will be selected on the first dimension
        
        
    indexes2: list of int, optional    
        gives the indexes that will be selected on the second dimension
        same as indexes if unassigned
   
    Returns
    =======
    
    extracted_array: numpy array

    """
    if not indexes2:
        indexes2 = indexes
        
    if array.ndim > 2:
        extracted_array = array[indexes,:,:]
        array = extracted_array.copy()
        extracted_array = array[:,indexes2,:]
    else:
#        extracted_array = array[indexes,:]
#        array = extracted_array.copy()
#        extracted_array = array[:,indexes2]     
        u = np.array([ indexes ])
        v = np.array( indexes )[:,np.newaxis] # transpose
        extracted_array = array[u,v]
        
    return extracted_array
    
  
    