# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 10:05:25 2013

@author: Salma Bougacha
"""
import sys
import matplotlib.pyplot as plt
from colorbar_test import MidpointNormalize


def plot_matrix(m,ax,ticks=[],tickLabels=[],title="",ylabel="",abs_max = None,
                abs_min = None, symmetric_cb = True, add_cb = True):
    """
    Plots matrix image with colorbar
    
    Parameters
    ==========
    m: np.array
        Matrix to plot
    ax: matplotlib.axes.AxesSubplot
        Current axes of the subplot
    ticks: list, optional
        Ticks on the x and y axis
    tickLabels: list, optional
        TickLabels on the y axis
    title: str, optional
        Title of the subplot
    ylabel: str, optional
        ylabel of the subplot
    abs_max: float, default to None
        maximal value for matplotlib.imshow
    abs_min: float, default to None
        maximal value for matplotlib.imshow
    symmetric_cb: bool, defalut to True
        symmetric colorbar or not
    add_cb: bool, defalut to True
        add colorbar or not
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    if abs_max is None:
        abs_max = np.abs(m).max()
    if abs_max <= sys.float_info.epsilon:
        abs_max = 0.1     
    if symmetric_cb:
        abs_min = -abs_max
    else:
        if abs_min is None:
            abs_min = m.min()    
            
    if False:
        if abs_max == 1:
            abs_max = abs(m).max()
        if abs_max <= sys.float_info.epsilon:
            abs_max = 0.1
        if symmetric_cb:
            abs_min = -abs_max
        else:
            if abs_min == -1:
                abs_min = m.min()
#    if math.isnan(abs_max):
#        abs_max = 1
    norm = MidpointNormalize(midpoint=0)              
    im = plt.imshow(m, cmap=plt.cm.RdBu_r, norm=norm, interpolation="nearest",
              vmin=abs_min, vmax=abs_max)
             
                 
#    plt.xticks([],[],visible = False)
    locs, labels = plt.xticks(ticks,tickLabels,size = 8)   #8    
    ax.xaxis.set_tick_params(size=0)
    ax.xaxis.tick_bottom()
#    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)

    plt.yticks(ticks,tickLabels,size = 8)   #8    
    ax.yaxis.set_tick_params(size=0)
    ax.yaxis.tick_left()

    plt.title(title)
    ax.autoscale(False)
    
    all_ntwk = True
 
    n_rois = m.shape[0]
    n_an = 6
    n_dmn = 5    
    n_wmn = n_rois - n_an - n_dmn
    
    if all_ntwk:
        plt.plot([0-0.5,n_rois],[n_wmn-0.5,n_wmn-0.5],'k')
        plt.plot([0-0.5,n_rois],[n_wmn+n_an-0.5,n_wmn+n_an-0.5],'k')
        plt.plot([n_wmn-0.5,n_wmn-0.5],[0-0.5,n_rois],'k')
        plt.plot([n_wmn+n_an-0.5,n_wmn+n_an-0.5],[0-0.5,n_rois],'k') 
    else:
        plt.plot([0-0.5,11],[6-0.5,6-0.5],'k')
        plt.plot([6-0.5,6-0.5],[0-0.5,11],'k')

    plt.hold(True) 
    
    if add_cb:
        # create an axes on the right side of ax. The width of cax will be 5%
        # of ax and the padding between cax and ax will be fixed at 0.05 inch.
        divider = make_axes_locatable(ax)
        
        cax = divider.append_axes("right", size="5%", pad=0.05)
        
        
           
        
        int_abs_max = round(abs_max,0)          
        r = 0
        while  int_abs_max == 0:
            r = r+2
            int_abs_max = round(abs_max,r)
        cb_label_size = max(1,8-r)
        if symmetric_cb:
            cbticks = [abs_min,0,abs_max]
        else:
            cbticks = [abs_min,abs_max]
        #    print '%f et %f pour %s' % (abs_max,int_abs_max, title)
        tickFormat = '%.1f'
        if r>2:
            tickFormat = '%e' 
        cb = plt.colorbar(
        im, cax=cax,ticks=cbticks,format = tickFormat)
        cb.ax.tick_params(labelsize=cb_label_size)
        cb.set_label(label = ylabel) 
    
import numpy as np    


def cov_to_corr(cov):
    """
    Computes correlation from covariance
    """    
    d = np.sqrt(np.diag(cov))
    corr = cov/d
    corr = corr/d[:, np.newaxis]
    return corr


def idx_template(list1,template):
    """
    Returns the indexes in a given list corresponding to a template list.
    
    Parameters
    ==========
    list1: list 
        List to compare to the template.
        
    ROItemplate: list
        Templae

    Returns
    =======
    indexes: list of int
        The indexes of elements in list1 existing in the template
    
    """
    
    indexes = [list1.index(elt) 
    for elt in template if list1.__contains__(elt)]
    return indexes



import os.path
import glob
from scipy import io

def read_signal_from_CONN(projectPath,conditionNumber,ntwk):
    """
    Reads the ROI names and the ROI signal for all subjects 
    from a specified CONN project and returns them respectively as
    a list and a list of numpy.ndarray.
    
    Parameters
    ==========    
    projectPath: str
        Path of the CONN project.
        
    conditionNumber: int
        Number of the condition to extract
        data is extracted from files
        'ROI_Subject*_Condition%03d.mat' % conditionNumber
    
    ntwk: list of lists
        len(ntwks) is the number of networks to include
        ntwk[nt] is a list of str specifying the names of the ROIs 
        to be included in the analysis if present
        
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
    
    folderName = 'results/preprocessing'
    pathNames =  os.path.join(projectPath,folderName,
    'ROI_Subject*_Condition%03d.mat' % conditionNumber)
    files = glob.iglob(pathNames)
    allROINames = []
    allROISignals = []    
    for nfile, file in enumerate(files):
        mat = io.loadmat(file) 
        allROISignals.append(mat['data']) #Matlab cell of length total number of rois, including GM,WM,CSF in 
        #the begining and effects at the end        
        if nfile == 0:
            allROINames = mat['names'].transpose()  
            nAllROIs = len(allROINames)        
            allROINames = [allROINames[roi][0][0][:] for roi in range(nAllROIs)]

    # Check if the Matlab cell is not empty and has the correct dimension
    if not len(allROISignals) or allROISignals[0].ndim != 2:
        raise ValueError('Empty signal or wrong signal dimensions!')
    
    # Select the effective ROIs and get their order as in template
    ROIIndexes = [idx_template(allROINames,ntwk) for ntwk in ntwks] 
    # indexes for each network in the list allROINames
    flatROIIndexes = [index for sublist in ROIIndexes for index in sublist]  # all indexes    
      
    nSamples = len(allROISignals[0][0][0]) 
    subjects = [];
    for subject, ROISignal in enumerate(allROISignals):   
        subjects.append(np.empty((nSamples,0)))
        for roi in flatROIIndexes:
            subjects[subject] = np.concatenate(
            (subjects[subject],ROISignal[0][roi][:,0][:,np.newaxis]),axis=1)
        subjects[subject] -= subjects[subject].mean(axis=0)
        subjects[subject] /= subjects[subject].std(axis=0)

    presentNtwk = [[allROINames[index] for index in ROIIndexes[nt]] 
    for nt in range(len(ntwks))] 

    return presentNtwk, subjects
    


if __name__== '__main__':

    from nilearn.group_sparse_covariance import GroupSparseCovarianceCV
    from sklearn.covariance import GraphLassoCV
    from scipy import linalg
    
    parentPath = '/volatile/new/salma/subject1to40'
    projectName = 'conn_servier2_1to40sub_RS1-Nback2-Nback3-RS2_Pl-D_1_1_1'
    projectPath = os.path.join(parentPath,projectName) # path of the CONN project
    
    # Define ROIs and networks
    AN = ['vIPS_big','pIPS_big','MT_big','FEF_big','RTPJ','RDLPFC'] # biyu's order
    DMN = ['AG_big','SFG_big','PCC','MPFC','FP'] # biyu's order
    #WMN = ['RT_1_1','LT_1_1','PL.cluster002_1_1','LFG_1_1',
    #       'SFG.cluster002_1_1','LPC.cluster001_1_1',
    #       'SPL.cluster002_1_1'] # main peaks in SPM.mat
    WMN = ['IPL','LMFG_peak1','CPL_peak1_peak3','LT']            
    ntwks = [WMN,AN,DMN]
    ntwksNames = ['WMN','AN','DMN']
    nConditions_displayed = 1 # number of displayed conditions
    for condition in range(nConditions_displayed):
    
        # read the ROIs names and signals for all subjects from the CONN project
        presentNtwk, subjects = read_signal_from_CONN(
        projectPath,condition+1,ntwks) 
    
                
    
        nSubjects = len(subjects)     
    
        gsc = GroupSparseCovarianceCV(max_iter=50, verbose=1)
        gsc.fit(subjects)
      
        gl = GraphLassoCV(verbose=True)
    
            
        n_displayed = 5  # number of subjects displayed
        ntwksDim =  map(len,presentNtwk)
        tickLabels = [ntName for nt, ntName in enumerate(ntwksNames) # names of present networks
        if ntwksDim[nt]>0]
        ntwksDim = np.array(ntwksDim)
        ntwksDim = ntwksDim[ntwksDim>0] # dimensions of present networks
        bt = np.zeros(ntwksDim.shape,ntwksDim.dtype)    
        bt[1:] = ntwksDim[:-1]
        ticks = np.cumsum(bt) + ntwksDim/2    
        ncols = 6 # number of columns in subplot
      
        
        # plot the covariances
        fig = plt.figure(figsize=(10, 7))
        plt.subplots_adjust(hspace=0.4)
        for n, subject in enumerate(subjects[:n_displayed]):
            nSamples = subject.shape[0]
            emp_cov = np.dot(subject.T, subject) / nSamples
            emp_prec = linalg.inv(emp_cov)
            gl.fit(subject) # Fit one graph lasso per subject
            matr = [("group-sparse, prec\n$\\alpha=%.2f$" % gsc.alpha_, gsc.precisions_[..., n]),
                    ("group-sparse, cov\n$\\alpha=%.2f$" % gsc.alpha_, gsc.covariances_[..., n]),
                    ("graph lasso\n prec\n$\\alpha=%.2f$" % gl.alpha_, gl.precision_),
                    ("graph lasso\n cov\n$\\alpha=%.2f$" % gl.alpha_, gl.covariance_),
                    ("empirical\n prec", emp_prec),
                    ("empirical\n cov", emp_cov)]
            for i, (name, this_matr) in enumerate(matr):
                plt.subplot(n_displayed, ncols, ncols * n + i+1)
                title = ''
                if n == 0:
                    title = name
                plot_matrix(this_matr,plt.gca(),ticks,tickLabels,title)
    
        
        
    
        # Fit one graph lasso for all subjects at once
        gl.fit(np.concatenate(subjects))
       
        
  
        
#        ncols = 5
#        fig = plt.figure(figsize=(10, 7))
#        plt.subplots_adjust(hspace=0.4)
#        plt.subplot(1, ncols,  1)
#        title = "group-sparse\n mean cov\n$\\alpha=%.2f$" % gsc.alpha_        
#        plot_matrix(gsc.covariances_.mean(axis=2),plt.gca(),ticks,tickLabels,title)
             
#        if False:     
#            plt.subplot(1, ncols,  2)
#            plot_matrix(gscCorr.mean(axis=2),plt.gca())
#            plt.title("group-sparse\n mean corr\n$\\alpha=%.2f$" % gsc.alpha_)    
#            plt.subplot(1, ncols,  3)
#            plot_matrix(glCorr.mean(axis=2),plt.gca())
#            plt.title("graph lasso\n mean corr")      
        
#        plt.subplot(1, ncols,  2)
#        title = "graph lasso\n all subjects\n prec\n$\\alpha=%.2f$" % gl.alpha_
#        plot_matrix(gl.precision_,plt.gca(),ticks,tickLabels,title)
#        plt.subplot(1, ncols,  3)
#        title = "graph lasso\n all subjects\n corr\n$\\alpha=%.2f$" % gl.alpha_
#        plot_matrix(cov_to_corr(gl.precision_),plt.gca(),ticks,tickLabels,title)
        
#        plt.show()
        
        # save the group-sparse and graph-lasso precision matrice        
#        outfolder = os.path.join(projectPath,'results','firstlevel','precisions')
#        outfile = os.path.join(outfolder,'groupSparse')      
#        np.save(outfile, gsc.precisions_)
#        outfile = os.path.join(outfolder,'graphLasso')      
#        np.save(outfile, gl.precisions_)


