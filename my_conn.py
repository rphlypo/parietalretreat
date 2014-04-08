# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 15:15:50 2014

@author: sb238920
TODO: improve plot_matrix by class 
"""
from __future__ import print_function

import os.path

import pylab
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, io

import conn_utilities as conn
from first_level import FC 
from second_level_new import Comparison, corr_to_Z
from compute_precision import plot_matrix

class MyConn:
    """
    Parameters
    ----------
    origin: str
        origin of data
        
    folder: u
        path of the project
        
    Attributes
    ----------
    `run_`: dict 
        keys: str, condition names
        values: numpy.arrays, shapes (n_subjects,n_samples, n_features) 
        input subjects. Each element along axis = 0 is a 2D array, whose 
        columns contain signals. Sample number can vary from condition 
        to condition, but all conditions must have the same number of 
        features.
    `ntwk_`: list of tuples (ntwk, rois)
        ntwk: str, network name
        rois: list of str, rois names
    """
    def __init__(self, origin='test', folder=None):
        self.origin = origin 
        self.folder = folder 
        
    def set_params(self,**kwargs):
        for param, val in kwargs.items():
            self.setattr(param,val)

    def get_params(self,deep=True):
        return {"origin"  : self.origin,
                "folder"  : self.folder}   
                
    def setup(self):
        """ defines setup data
        Returns
        -------
        self.n_subjects: int
            number of subjects
            
        self.n_runs: int
            number of conditions. If not the same for all subjects, then it's
            the maximal number of conditions available for at least one 
            subject.
            
        self.covariates: dict
            Second level covariates.            
            Keys: str, covariate names
            Values: int or float, covariate values
        """
        if self.origin == 'from_conn':
            conn_x = io.loadmat(self.folder+'.mat', struct_as_record=False, 
                              squeeze_me=True)
            print(conn_x.Setup)
            self.n_subjects = conn_x.Setup.nsubjects
            self.n_runs = conn_x.Setup.nsessions.max()
            cov_names = conn_x.Setup.l2covariates.names
            cov_values = conn_x.Setup.l2covariates.values
            self.covariates  = {}
            for n, name in enumerate(cov_names):
                if name != ' ':
                    self.covariates [str(name)] = np.array(
                    [int(value[n]) for value in cov_values])

        return self
        
    def preproc(self, template_network):
        """
        Returns
        -------
        self.runs_: dict
            signals for each condition across all subject and defined network
        """
        self.conditions = []            
        self.runs_ = {}
        for n_run in range(self.n_runs):            
            for n_subject in range(self.n_subjects):
                file_path =  os.path.join(
                self.conn_folder,'results/preprocessing',
                'ROI_Subject%03d_Condition%03d.mat' %(n_subject+1, n_run+1))             
                if n_run == 0 and n_subject == 0: # same ROIs in all files                
                    self.ntwk_ = conn.network(file_path, template_network)
                    rois = [roi for tup in self.ntwk_ for roi in tup[1]] 
                        
                if n_subject == 0:         
                    condition_name = conn.cond_name(file_path)
                    self.conditions.append(condition_name)
                    print(condition_name, end=', ')
                    self.runs_[condition_name] = []
                     
                subject = conn.extract_rois(file_path, rois)     
                self.runs_[condition_name].append(subject)
        return self     
    
    def analysis(self, template_network, standardize = False, *args):        
        """Extracts rois time series for the template rois and computes
        connectivity values for the chosen measure(s).
        
        Parameters
        ----------
        standardize: bool (optional, default to False)
            standardize roi signals or not
        template_network: list of tuples (ntwk, rois)
            ntwk: str, network name
            rois: list of str, rois names
        *args: optional str, default to "correlations"
            names of the connectivity measures.
        """
        n_runs, n_subjects = conn.count_preproc(self.conn_folder)
        print('{} runs, {} subjects'.format(n_runs, n_subjects))
        print("conditions: ",end='')
        self.conditions = []            
        self.runs_ = {}
        self.measure_names = args
        self.fc_ = {}
        for n_run in range(n_runs):            
            for n_subject in range(n_subjects):
                file_path =  os.path.join(
                self.conn_folder,'results/preprocessing',
                'ROI_Subject%03d_Condition%03d.mat' %(n_subject+1, n_run+1))             
                if n_run == 0 and n_subject == 0: # same ROIs in all files                
                    self.ntwk_ = conn.network(file_path, template_network)
                    rois = [roi for tup in self.ntwk_ for roi in tup[1]]                   
                    if 'segregations' in args:
                        ntwk_dims =  map(len,self.ntwk_)                    
                    else:                    
                        ntwk_dims = None
                        
                if n_subject == 0:         
                    condition_name = conn.cond_name(file_path)
                    self.conditions.append(condition_name)
                    print(condition_name, end=', ')
                    self.runs_[condition_name] = []
                    fcs = []
                     
                subject = conn.extract_rois(file_path, rois)     
                self.runs_[condition_name].append(subject)
                myFC = FC(subject, standardize, ntwk_dims)            
                myFC.compute(*args)
                for n_measure, measure_name in enumerate(args):
                    if n_subject == 0:
                        n_features = myFC.conn[measure_name].shape[0]
                        fcs.append(np.empty((n_subjects,n_features,n_features)))
                    
                    fcs[n_measure][n_subject] = myFC.conn[measure_name]
            for n_measure, measure_name in enumerate(args):
                self.fc_[(condition_name,measure_name)] = fcs[n_measure]
            # self.fc_[condition_name] = zip(args, fcs)               
        print('\ncomputed measures: ',end='')
        print(*args,sep=', ')
        return self     
    
#    def visualize:
        
        
    def analysis_fig(self, fig_dir=None, overwrite=None, n_subjects=1, cond_names=None, 
                     measure_names=None):
        if fig_dir is None:
            fig_dir = os.getcwd()           
        elif not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
            
        if cond_names is None:
            cond_names = self.conditions
        elif set(cond_names) - set(self.conditions):
           ValueError("Unkown condition") 
           
        n_conds = len(cond_names)   
        if measure_names is None:
            measure_names = self.measure_names         
        elif set(measure_names) - set(self.measure_names):
           ValueError("Unkown measure")             
            
        ntwk_dims = np.array([len(tup[1]) for tup in self.ntwk_]) # number of rois in each network
        ntwk_names = [ntwk for ntwk, rois in self.ntwk_]
        roi_names = [roi.split('_',1)[0] for ntwk, rois in self.ntwk_ for roi in rois]
        for measure_name in measure_names:
            if measure_name == 'segregations':
                measure_name_out = 'integration, mutual information'
            else:
                measure_name_out = measure_name
                
            plt.figure(figsize=(10, 10)) #(10,7) to reshape dynamically
            plt.subplots_adjust(hspace=0.4,wspace=0.5) # hspace=0.4
            fcs = np.array([self.fc_[(cond_name,measure_name)][:n_subjects]
            for cond_name in cond_names])            
            if fcs.shape[2] == len(self.ntwk_): # network level measure
                ticks = range(len(self.ntwk_))
                symmetric_cb = False
                labels = ntwk_names 
            else:
                tmp = np.zeros(ntwk_dims.shape)   
                tmp[1:] = ntwk_dims[:-1]
                ticks_ntwk = np.cumsum(tmp) + ntwk_dims/2 # for undetailed ROI measures
                ticks = range(len(roi_names))#ticks = ticks_roi
                maximum = 1
                minimum = -1
                symmetric_cb = True   
                labels= roi_names                
                
            maximum = fcs.max()
            minimum = fcs.min()
            for n_subject in range(n_subjects):
                for n_cond, cond_name in enumerate(cond_names):
                    plt.subplot(n_subjects, n_conds,  n_conds * n_subject +n_cond+1)
                    title = ''
                    if n_conds > 3:                  
                        add_cb = False
                        ylabel = ''
                        tickLabels = []
                    else:
                        add_cb = True
                        ylabel = measure_name_out
                        tickLabels = labels
                        
                    if n_subject == 0:
                        title = cond_name    
                        
                    if n_cond == n_conds-1:    
                        ylabel = measure_name_out    
                        add_cb = True
                    elif n_cond == 0:
                        tickLabels = labels
                    
                    ylabel = ''
                    title = ''
                    plot_matrix(fcs[n_cond,n_subject],plt.gca(),ticks,tickLabels,
                                title, ylabel,maximum,minimum, symmetric_cb, add_cb)      
     
            plt.draw()  # Draws, but does not block
            # raw_input() # waiting for entry
            fig_title = measure_name.replace(" ", "_")
            filename = fig_dir + fig_title + ".pdf"   
            if not os.path.isfile(filename) or overwrite:
                pylab.savefig(filename) 
                os.system("pdfcrop %s %s" % (filename, filename))    
            
        plt.show()            
        

        
    def results(self, tests, corrected=False): # todo: add masking option
        if self.conn_folder:
            print("Reading covariates from CONN_x structure ...", end = ' ')
            covs = conn.covariates(self.conn_folder)        
            print("covariates ok.")

            #cond_names = conn.conditions(self.conn_folder)      
#        else:
#            cov_names = fake_covs
#            cond_names = fake_conds
            
#        self.results_ = []     
        cond_names = self.conditions
        identity = lambda x: x
        transforms = {'correlations':corr_to_Z,
                      'partial correlations':corr_to_Z,
                      'semi-partial correlations':corr_to_Z,
                      'segregations':identity}
        self.results_ = {}                    
        for the_cond_names, the_cov_names in tests:
            if not set(the_cond_names).issubset(cond_names):
                print('One or more conditions named', end=' ')
                print(*the_cond_names,sep=', ', end=' ')
                print('do not exit.')
                continue
            
            if not set(the_cov_names).issubset(covs.keys()):
                print('One or more covariate named', end=' ')
                print(*the_cov_names,sep=', ', end=' ')
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
                self.results_[(test_name,measure_name)] = comp.stats_difference_
            print(test_name+' compared.')                
        return self
        

def identity(x):
    return x
    
    
def fc_func(*args):
    """Makes the function compute_fc that computes connectivity.
    Only the needed computations are done depending on the specified measures.
        
    Parameters
    ===========    
    *args: optional, str
        name(s) of the connectivity measures to use.
    Returns
    =======
    compute_fc: function
        adequate function to compute the connectivity values for the 
        specified measures.
    """    
    fc_func = {'correlations': compute_corr,
               'partial correlations': compute_partial,
               'segregations': compute_seg} 
    def compute_fc(subjects, standardize = True, ntwks_dims = None):
        """ Functional connectivity matrices for specified measures.
        
        Parameters
        ==========
        subject: array, shape(n_smaples,n_features)
            roi times series
        standardize: bool, optional, default to True
            standardize or not signal for each roi
        measure_names: list of str
            name(s) of connectivity measure(s)
        ntwks_dims: list of int, optional
            number of rois in each network
    
        Returns
        ========
        fc: dict
            keys: str, names of the connectivity measures
            values: arrays of shape (n_features,n_features), connectivity values.
        """
        for subject in subjects:
            subject -= subject.mean(axis=0)     
            if standardize:
                subject = subject/subject.std(axis=0) # copy on purpose
            
            n_samples = subject.shape[0]
            cov = np.dot(subject.T, subject) / n_samples # covariance matrix  
            fc = {} 
            
    #        if not args:
    #            args = ("correlations")
               
            if set(args).intersection(set(['correlations','variability'])):
                fc['correlations'] = cov_to_corr(cov)
    
            if set(args).intersection(set(['partial correlations','segregations'])):
                cond_number = np.linalg.cond(cov) 
                if cond_number > 100:#1/sys.float_info.epsilon:
                    print('Bad conditioning! condition number is {}'.format(cond_number))
                prec = linalg.inv(cov) # precision matrix  
                if 'partial correlations' in args:            
                    fc['partial correlations'] = prec_to_partial_corr(prec) # partial correlation matrix            
                if 'segregations' in args:
                    integ = compute_integration(prec,np.array(ntwks_dims)) # integration matrices     
                    fc['segregations'] = compute_segregation(integ)  # segregation matrices 
                    
            return fc
    return compute_fc
            




