# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 14:47:32 2014

@author: sb238920
"""



if False:
    gsc = GroupSparseCovarianceCV(max_iter=50, verbose=1)
    gsc.fit(subjects)            
    emp_covs[:,n_condition,...] = np.reshape(
    gsc.covariances_,(n_subjects,n_rois,n_rois))
    emp_precs[:,n_condition,...] = np.reshape(
    gsc.precisions_,(n_subjects,n_rois,n_rois))
    time.sleep(5)
    for n_subject in range(n_subjects):
        cond_number = np.linalg.cond(emp_covs[n_subject,n_condition,...])
        invert = abs(np.dot(emp_covs[n_subject,n_condition,...],
                         emp_precs[n_subject,n_condition,...],) - np.identity(15)).max()
        if cond_number > 100:
            print "max sparse_cov x sparse_prec= %g cond_number = %f for subject %d condition %d" %(invert,cond_number,n_subject,n_condition) 
            time.sleep(1)                         
