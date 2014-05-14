# Standard library imports
import random
import os
import sys
import re

# Related third party imports
import numpy as np
from numpy.testing import assert_almost_equal
from scipy import io
from pandas import DataFrame

# Local application/library specific imports
import classify_covs as my_classif
from manifold import random_spd
from connectivity import cov_to_corr
sys.path.append("/home/sb238920/CODE/servier2")
from conn_utilities import get_structurals, get_conditions, extract_rois


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


def test_statistical_test():
    """Testing function statistical_test"""
    # Prepare the dataframe
    conn_file = "/volatile/new/salma/subject1to40/" + \
    "conn_servier2_1to40sub_RS1-Nback2-Nback3-RS2_Pl-D_1_1_1.mat"
    if not os.path.isfile(conn_file):
        raise IOError("file {0} not found".format(conn_file))

    conn_class = io.loadmat(conn_file, struct_as_record=False,
                            squeeze_me=True)['CONN_x']
    structurals = get_structurals(conn_class)
    subjects = [re.findall("([A-Z]{2}[0-9]{6})", file_name)[0] for file_name in
        structurals]
    conditions = get_conditions(conn_class)
    rois = ['IPL', 'LMFG_peak1', 'RCPL_peak1', 'LCPL_peak3', 'LT',
            'vIPS_big', 'pIPS_big', 'MT_big', 'FEF_big', 'RTPJ', 'RDLPFC',
            'AG_big', 'SFG_big', 'PCC', 'MPFC', 'FP']
    conn_folder = os.path.splitext(conn_file)[0]
    df_list = list()
    for s, subject in enumerate(subjects):
        for c, condition in enumerate(conditions):
            file_name = "ROI_Subject%03d_Condition%03d.mat" % (s + 1, c + 1)
            regions_signal = extract_rois(os.path.join(
                conn_folder, "results/preprocessing", file_name), rois)
            df_list.append({"condition": condition,
                            "subj_id": subject,
                            "region_signals": regions_signal})
    df = DataFrame(df_list)

    # Launch statistics
    t_test = my_classif.statistical_test(df, conditions, p_correction="fdr",
                                         estimators={'kind': 'tangent',
                                                     'cov_estimator': None})
    # Visualize results