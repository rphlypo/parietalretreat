# -*- coding: utf-8 -*-
"""
Test on servier dataset
"""
import sys
import os

import numpy as np
import matplotlib.pylab as plt

from connectivity import CovEmbedding, vec_to_sym, sym_to_vec
sys.path.append('/home/sb238920/CODE/servier2')
from my_conn import MyConn
from compute_precision import plot_matrix

 # biyu's order
AN = ['vIPS_big', 'pIPS_big', 'MT_big', 'FEF_big', 'RTPJ', 'RDLPFC']
DMN = ['AG_big', 'SFG_big', 'PCC', 'MPFC', 'FP']
final = 1
if final:
#        WMN = ['IPL','MFG_peak1_peak2','LMFG_peak1','RMFG_peak2',
#              'CPL_peak1_peak3','RCPL_peak1','RCPL_peak2','LCPL_peak3','LT']
    #WMN = ['IPL','LMFG_peak1','CPL_peak1_peak3','LT']
    WMN = ['IPL', 'LMFG_peak1',  # 'RMFG_peak2' removed because of bad results
           'RCPL_peak1', 'LCPL_peak3', 'LT']
else:
    WMN = ['RT', 'LT', 'PL.cluster002', 'LFG',
           'MFG.cluster002', 'LPC.cluster001',
           'SPL.cluster002']  # main peaks in SPM.mat

template_ntwk = [('WMN', WMN), ('AN', AN), ('DMN', DMN)]
conn_folder = "/volatile/new/salma/subject1to40/conn_servier2_1to40sub_RS1-Nback2-Nback3-RS2_Pl-D_1_1_1"
mc = MyConn('from_conn', conn_folder)
mc.setup()
#fake_mc = MyConn(setup='test')
standardize = True
mc.analysis(template_ntwk, standardize, 'correlations', 'partial correlations',
            'segregations', 'precisions')  # TODO include variablity

rs1_pl = "ReSt1_Placebo"
nb2_pl = "Nbac2_Placebo"
nb3_pl = "Nbac3_Placebo"
rs2_pl = "ReSt2_Placebo"
rs1_d = "ReSt1_Drug"
nb2_d = "Nbac2_Drug"
nb3_d = "Nbac3_Drug"
rs2_d = "ReSt2_Drug"
conditions = [(rs1_pl, nb2_pl), (rs1_pl, nb3_pl), (rs2_pl, nb2_pl),
              (rs2_pl, nb3_pl), (rs1_pl, rs1_d), (nb2_pl, nb2_d),
              (nb2_pl, nb2_d), (rs2_pl, rs2_d)]
conditions = [(rs1_pl, nb2_pl)]
# Between subjects covariates
n_subjects_A1 = 21
n_subjects_A2 = 19
n_subjects = 40
group_all = np.ones((1, n_subjects))
group_A1 = np.hstack(
(np.ones((1, n_subjects_A1)), np.zeros((1, n_subjects_A2))))
group_A2 = np.hstack(
(np.zeros((1, n_subjects_A1)), np.ones((1, n_subjects_A2))))
groups = {'All': group_all,
          'A1': group_A1,
          'A2': group_A2}
for kind in ["correlation", "tangent", "precision", "partial correlation"]:
    estimators = {'kind': kind, 'cov_estimator': None}
    for condition1, condition2 in conditions:
        print(condition1, condition2)
        signals1 = mc.runs_[condition1]
        signals2 = mc.runs_[condition2]
        signals_list = [subj for subj in signals1] + [subj for subj in signals2]
        cov_embedding = CovEmbedding(**estimators)
        X = cov_embedding.fit_transform(signals_list)
        print estimators['kind']
        #fre_mean = cov_embedding.mean_cov_
        nsubjects = X.shape[0] / 2
        X = [vec_to_sym(x) for x in X]
        X = np.asarray(X)
        mc.fc_[(condition1, estimators['kind'])] = X[:nsubjects, :]
        mc.fc_[(condition2, estimators['kind'])] = X[nsubjects:, :]
        mc.measure_names = mc.measure_names + (estimators['kind'], )
        fig_dir = "/home/sb238920/slides/servierFinal/Images_corrected/"    
    #    mc.analysis_fig(fig_dir, 'overwrite',
    #                n_subjects=3, cond_names=["ReSt1_Placebo", "Nbac3_Placebo"],
    #                measure_names=["tangent"])
        paired_tests = [([condition1, condition2], ['All'])]
        mc.results(paired_tests, corrected=True)  # add masking option

        # Plotting second level result, TODO: encapsulate in second_level_new
        test_name = "All " + condition1 + " vs All " + condition2
        measure_name = estimators['kind']
        comp = mc.results_[(test_name, measure_name)]
        th = 0.05
        comp.threshold(th)
        test_names = [test_name]
        measure_names = [measure_name]
        ticks = []
        tickLabels = []
        titles = [condition2, condition1, "difference"]
        ylabels = [kind, kind, ""]
        abs_maxs = [None, None, None]
        abs_mins = [None, None, None]
        symmetric_cbs = [True, True, True]
        fig = plt.figure(figsize=(10, 7))
        plt.subplots_adjust(hspace=0.8, wspace=0.5)
        signifs = [comp.signif_follow_up_,
                   comp.signif_baseline_,
                   comp.signif_difference_] #vec_to_sym(comp.signif_difference_)
        matr = zip(titles, signifs)
        for i, (name, this_matr) in enumerate(matr):
            plt.subplot(1, 3, i + 1)
            plot_matrix(this_matr, plt.gca(), title = name, ticks=ticks, tickLabels=tickLabels,
                        abs_max=abs_maxs[i], abs_min=abs_mins[i],
                        symmetric_cb=symmetric_cbs[i], ylabel = ylabels[i])
        plt.draw()  # Draws, but does not block
        # raw_input() # waiting for entry
        fig_title = test_name + "_" + measure_name
        fig_title = fig_title.replace(" ", "_")
        filename = fig_dir + fig_title + ".pdf"
        overwrite = True
        if not os.path.isfile(filename) or overwrite:
            plt.savefig(filename)
            os.system("pdfcrop %s %s" % (filename, filename))

plt.show()


#    mc.results_fig(th, fig_dir, "overwrite", test_names,
#                 measure_names)
#    mc.performances(perfs_file, test)
#    overwrite = True
#    mc.analysis_fig = (overwrite,'/home/sb238920/slides/servier2/Images')

