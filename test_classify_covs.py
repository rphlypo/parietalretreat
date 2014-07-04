# Standard library imports
import random
import os
import nose

# Related third party imports
import numpy as np
from numpy.testing import assert_almost_equal

# Local application/library specific imports
import classify_covs as my_classif
from matrices_generator import random_spd
from connectivity import cov_to_corr


def test_corr_to_Z():
    """Testing function corr_to_Z"""
    n_subjects = random.randint(2, 60)
    shape = random.randint(2, 100)
    input_corrs = np.empty((n_subjects, shape, shape))
    for input_corr in input_corrs:
        input_corr = random_spd(shape, shape)
        input_corr = cov_to_corr(input_corr)
    Z = my_classif.corr_to_Z(input_corrs)
    output_corrs = np.tanh(Z)
    assert_almost_equal(output_corrs, input_corrs)


def unit_statistical_conn(conn_folder, subjects_pattern, rois=None,
                          exclude_masks=True, p_correction=None,
                          estimator_names=None, kinds=None, p_th=0.05,
                          overwrite=False):
    """Testing statistical_test on output from a given CONN project"""
    # TODO: include diff tests
    df = my_classif.create_df_from_conn(conn_folder, subjects_pattern,
                                        rois=rois)
    stat_data = my_classif.create_statistical_df(df, p_correction=p_correction,
                                      estimator_names=estimator_names,
                                      kinds=kinds)
    base_fig_dir = conn_folder
    if p_correction is None:
        correction_dir = "uncorrected"
    else:
        correction_dir = "corrected"
    for stat_df, estimator_name, kind in stat_data:
        fig_dir = os.path.join(base_fig_dir, estimator_name, correction_dir,
                               kind)
        my_classif.visualize_statistical_df(stat_df, estimator_name, kind,
                                            fig_dir, comparisons=None,
                                            p_th=p_th, overwrite=overwrite)

     # Visualize individual connectivity matrices
#    plt.figure(figsize=(10, 7))
#    matrices = [vec_to_sym(signif, isometry=False) for signif in
#        [signif_f, signif_b, signif]]
#    names = [f, b, comp]
#    for i, matr in enumerate(matrices):
#        plt.subplot(1, 3, 1 + i)
#        plot_matrix(matr, plt.gca(), title=names[i], ticks=[],
#                    tickLabels=[], symmetric_cb=True,
#                    ylabel="significative mean")
#        plt.draw()


def test_statistical():
    """Testing function statistical_test on different CONN projects"""
    # Servier dataset
    conn_folder = "/volatile/new/salma/subject1to40/" + \
    "conn_servier2_1to40sub_RS1-Nback2-Nback3-RS2_Pl-D_1_1_1"
    subjects_pattern = "([A-Z]{2}[0-9]{6})"
    rois = ['IPL', 'LMFG_peak1', 'RCPL_peak1', 'LCPL_peak3', 'LT',
            'vIPS_big', 'pIPS_big', 'MT_big', 'FEF_big', 'RTPJ', 'RDLPFC',
            'AG_big', 'SFG_big', 'PCC', 'MPFC', 'FP']

    unit_statistical_conn(conn_folder, subjects_pattern, rois=rois,
                          exclude_masks=True, p_correction=None,
                          estimator_names=["empirical"], kinds=["correlation"],
                          overwrite=True, p_th = 1.)

    # ds107 dataset
    conn_folder = os.path.join("/media/Elements/workspace/brainpedia/preproc/",
                               "salma/gunzipped_ds107/conn_study")
    subjects_pattern = "([a-z]{3}[0-9]{3})"
    rois = ['vPVC', 'dPVC',
            'vIPS', 'pIPS', 'MT', 'FEF', 'RTPJ', 'RDLPFC',
            'AG', 'SFG', 'PCC', 'MPFC', 'FP']

#    unit_statistical_conn(conn_folder, subjects_pattern, rois=rois,
#                          exclude_masks=True, p_correction=None,
#                          estimator_names=["empirical"], kinds=["correlation"],
#                          overwrite=True)
    print stop

if __name__ == "__main__":
    nose.run()