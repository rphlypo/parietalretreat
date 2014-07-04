import sys
import copy
import os.path
import re
import warnings

import numpy as np
import scipy.linalg
import scipy.stats.mstats
import matplotlib.pylab as plt
from scipy import io
from pandas import DataFrame

from sklearn.covariance import LedoitWolf, ShrunkCovariance

import setup_data_paths
import confound
import show_connectomes
import pval_correction
from connectivity import CovEmbedding, vec_to_sym, sym_to_vec
sys.path.append("/home/sb238920/CODE/servier2")
from conn_utilities import get_structurals, get_conditions, get_rois,\
    extract_rois
from compute_precision import plot_matrix


def create_df_from_conn(conn_folder, subjects_pattern, rois=None,
                    exclude_masks=True):
    """Returns a DataFrame based on data from a conn project.

    Parameters
    ==========
    conn_folder: str
        path to the conn project folder.
    subject_pattern: str
        pattern for subjects identifiers in the structural paths
    rois: list of str, optional
        list of regions labels. If None, all the present regions are labeled in
        the same order as in the conn project.
    exclude_masks: boolean, optional
        If True, excludes the 'Grey Matter', 'White Matter' and 'CSF' regions
        from rois.

    Returns
    =======
    df: class 'pandas.core.frame.DataFrame'
        The output data frame. Columns are 'subj_id' for the subjects
        idenifiers, 'conditions' for the conditions names and 'regions_signal'
        for the preprocessed time series within each labeled region.
    """
    if not os.path.isdir(conn_folder):
        raise IOError("folder {0} not found".format(conn_folder))

    conn_file = conn_folder + ".mat"
    if not os.path.isfile(conn_file):
        raise IOError("file {0} not found".format(conn_file))

    conn_class = io.loadmat(conn_file, struct_as_record=False,
                            squeeze_me=True)['CONN_x']
    structurals = get_structurals(conn_class)
    subjects = []
    for file_name in structurals:
        subj_id = re.findall(subjects_pattern, file_name)[0]
        if subj_id:
            subjects.append(subj_id)
        else:
            raise ValueError("no match for pattern {0} in structural path {1}",
                             subjects_pattern, file_name)

    all_rois = get_rois(conn_class)
    if rois is None:
        rois = all_rois

    if not set(rois).issubset(set(all_rois)):
        warnings.warns("No data for region(s) {0} in the conn project,"
            "removing it".format(set(rois).discard(set(rois).intersection(
            set(all_rois)))))

    rois = [roi for roi in rois if roi in all_rois]

    if exclude_masks:
        for mask in ['Grey Matter', 'White Matter', 'CSF']:
            if mask in rois:
                rois.remove(mask)

    conditions = get_conditions(conn_class)
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
    return df

# TODO: function to visualize first level results

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


def get_data(root_dir="/",
             data_set=None,
             maps_img=None,
             **kwargs):
    if data_set is None:
        data_set = "ds107"
    df, region_signals = _load_data(root_dir=root_dir,
                                    data_set=data_set,
                                    maps_img=maps_img,
                                    **kwargs)
#    print [os.path.split(df["anat"][k]) for k in xrange(df["anat"].shape[0])]
    return _get_region_signals(df, region_signals, data_set=data_set)


def _load_data(root_dir="/",
               data_set="ds107", maps_img=None,
               cache_dir="/volatile/storage/workspace/parietal_retreat/" +
               "covariance_learn/cache/",
               n_jobs=1):
    from joblib import Memory
    mem = Memory(cachedir=cache_dir)
    load_data_ = mem.cache(setup_data_paths.run)

    df = setup_data_paths.get_all_paths(root_dir=root_dir, data_set=data_set)
    # region_signals = joblib.load(os.path.join(root_dir, dump_file))
    region_signals = load_data_(root_dir=root_dir, data_set=data_set,
                                n_jobs=n_jobs, maps_img=maps_img,
                                dump_dir=os.path.join(cache_dir, data_set))
    return df, region_signals


def _get_conditions(root_dir, data_set="ds107"):
    data_set_dir = os.path.join(root_dir,
                                data_set,
                                "models/model001/condition_key.txt")

    with open(data_set_dir) as f:
        conditions = list()
        while True:
            try:
                line = f.readline()
                if not line:
                    raise StopIteration
                if len(line.split()) > 3:
                    conditions.append(" ".join(line.split()[2:]))
                else:
                    conditions.append(line.split()[2])
            except StopIteration:
                return conditions


def _get_region_signals(df, region_signals, data_set="ds107"):
    df_ = df.groupby(["condition", "subj_id"])
    df_list = list()
    for names, group in df_:
        data = list()
        for ix_ in range(len(group)):
            onset_file, TR, region_ix, confd_file =\
                [group.iloc[ix_][k]
                 for k in ["cond_onsets", "TR", "region_ix", "confds"]]
            confds = confound.compute_mvt_confounds(confd_file)[0]
            signals = _regress(region_signals[region_ix], confds)
            data.append(_get_samples(signals, onset_file, TR))
        arr = np.vstack(data)
        df_list.append({"condition": names[0],
                        "subj_id": names[1],
                        "region_signals": arr})
    return DataFrame(df_list)


def _get_samples(signals, onset_file, TR):
    onsets = np.loadtxt(onset_file)
    onsets_ = onsets[..., 0]
    X1 = np.hstack((onsets_[:-1][np.diff(onsets_) > TR], onsets_[-1]))
    X2 = np.hstack((onsets_[0], onsets_[1:][np.diff(onsets_) > TR]))

    y1 = np.ones(shape=(X1.shape[0], ))
    y2 = np.zeros(shape=(X2.shape[0], ))

    X = np.hstack((X1, X2))
    y = np.hstack((y1, y2))

    y = y[np.argsort(X, kind="mergesort")]
    X = np.sort(X, kind="mergesort") + 5.

    z = np.arange(0, signals.shape[0] * np.float(TR), np.float(TR))

    ix_ = np.array([True
                    if np.any(np.logical_and(X2 < t, t <= X1)) else False
                    for t in z])

    return signals[ix_, ...] - np.mean(signals[ix_, ...], axis=0)


def _get_symm_psd_mx(df, CovEst):
    covs = list()
    for ix_ in range(len(df)):
        time_series = df.iloc[ix_]["region_signals"]
        cov_est = CovEst(assume_centered=True)
        covs.append(cov_est.fit(time_series).covariance_)
    df["covs"] = covs
    return df


def _regress(X, y):
    Q, _ = scipy.linalg.qr(y, mode="economic")
    return X - Q.dot(np.linalg.pinv(Q.T.dot(Q))).dot(Q.T.dot(X))


def corr_to_Z(corr):
    """
    Gives the Z-Fisher transformed correlation matrix. Correlations 1 and -1
    are transformed to nan.

    Parameters
    ==========
    corr: np.array
        correlation matrix

    Returns
    =======
    Z: np.array
        Z-Fisher transformed correlation matrix
    """
    eps = sys.float_info.epsilon  # 1/1e9
    Z = copy.copy(corr)           # to avoid side effects
    corr_is_one = 1.0 - abs(corr) < eps
    Z[corr_is_one] = np.inf * np.sign(Z[corr_is_one])
    #0.5*np.log((1+corr[1.0 - corr >= eps])/(1-corr[1.0 - corr >= eps]))
    Z[np.logical_not(corr_is_one)] = \
        np.arctanh(corr[np.logical_not(corr_is_one)])
    return Z


def var_stabilize(X, kind):
    """Apply to each entry of array the variance stabilizing transform

    Parameters
    ==========
    X: array
        input data

    kind: str
        covariance embedding kind

    Returns
    =======
    Y: array
        transformed data, same shape as Y
    """
    if kind in ['correlation', 'partial correlation']:
        Y = corr_to_Z(X)
    else:
        Y = X

    return Y


def var_unstabilize(X, kind):
    """Apply to each entry of array the variance stabilizing transform

    Parameters
    ==========
    X: array
        input data

    kind: str
        covariance embedding kind

    Returns
    =======
    Y: array
        transformed data, same shape as Y
    """
    if kind in ['correlation', 'partial correlation']:
        Y = np.tanh(X)
    else:
        Y = X

    return Y


def statistical_test(df, conditions, estimators={'kind': 'tangent',
                                                 'cov_estimator': None},
                     p_correction=None,
                     n_jobs=1):  # TODO threshold when plot
    grouped = df.groupby(["condition", "subj_id"])
    dict_list = list()
    entries = ("baseline", "pval baseline", "mean baseline",
               "follow up", "pval follow up", "mean follow up",
               "comparison", "tstat", "pval", "mean")
    for (ix1_, condition1) in enumerate(conditions):
        for (ix2_, condition2) in enumerate(conditions):
            if ix1_ <= ix2_:
                continue
            cond = list()
            grouped = df.groupby("subj_id")
            for _, group in grouped:
                cond.append(group[group["condition"] == condition1]
                            ["region_signals"].iloc[0])
                cond.append(group[group["condition"] == condition2]
                            ["region_signals"].iloc[0])
            X = CovEmbedding(**estimators).fit_transform(cond)
            X = [vec_to_sym(x) for x in X]
            X = np.asarray(X)
            X = sym_to_vec(X, isometry=False)
            Y = var_stabilize(X, estimators['kind'])
            t_stat_baseline, p_baseline = scipy.stats.ttest_1samp(Y[::2, ...],
                                              0.0,
                                              axis=0)
            q_baseline = pval_correction.correct(p_baseline,
                                                 correction=p_correction)
            q_baseline[np.isnan(q_baseline)] = 0.
            baseline_mean = var_unstabilize(
                Y[::2, ...], estimators['kind']).mean(axis=0)
            t_stat_followup, p_followup = scipy.stats.ttest_1samp(Y[1::2, ...],
                                              0.0,
                                              axis=0)
            q_followup = pval_correction.correct(p_followup,
                                                 correction=p_correction)
            q_followup[np.isnan(q_followup)] = 0.
            followup_mean = var_unstabilize(
                Y[1::2, ...], estimators['kind']).mean(axis=0)
            t_stat, p = scipy.stats.ttest_rel(Y[::2, ...],
                                              Y[1::2, ...],
                                              axis=0)
            q = pval_correction.correct(p, correction=p_correction)
            #q[np.isnan(q)] = 0.
            comp_mean = (var_unstabilize(Y[1::2, ...], estimators['kind']) -\
                var_unstabilize(Y[::2, ...], estimators['kind'])).mean(axis=0)
#            print "{} vs. {}: t_stat = {}, q-val = {}".format(
#                condition1, condition2, t_stat, q)
            dict_list.append(
                dict(zip(*[entries,
                           ("{}".format(condition1),
                            q_baseline, baseline_mean,
                            "{}".format(condition2),
                            q_followup, followup_mean,
                            "{} vs. {}".format(condition1, condition2),
                            t_stat, q, comp_mean)])))
    return DataFrame(dict_list, columns=entries)


def plot_results(t_df, save_dir, p_th=.05, estim_title=None, labels=None):  # TODO threshold when plot
    if estim_title is not None:
        estim_title = "{}".format(estim_title)
    else:
        estim_title = ""
    for ix_ in range(len(t_df)):
#        tstats[pvals > p_th] = 0.
        names = ["baseline", "follow up", "comparison"]
        titles = [t_df[name].iloc[ix_].replace(" ", "_") + "_" + estim_title
            for name in names]
        matrices = [vec_to_sym(t_df["mean " + name].iloc[ix_],
                               isometry=False) for name in names]
        for matrix, title in zip(matrices, titles):
            show_connectomes.plot_adjacency(matrix, n_clusters=1,
                                            title=title, labels=labels,
                                            vmin=None, vmax=None,
                                            col_map="red_blue_r",
                                            fig_name=os.path.join(
                                            save_dir, title + ".pdf"))


def create_statistical_df(df, p_correction=None, estimator_names=None,
                          kinds=None):
    """Computes the second level data frame.

    Parameters
    ==========
    estimator_names: list of str
        Estimators names. Possible elements are "empirical", "shrunk" and
        "ledoit". If None, all elements are included.

    kinds: list of str
        Connectivity measures. Possible elements are "correlation",
        "precision", "partial correlation" and "tangent". If None, all elements
        are included.

    Retruns
    =======
    stat_data: list of tuples (class 'pandas.core.frame.DataFrame', str, str)
        The list of statistical DataFrames, etimator name and kind.
    """
    cov_estimator = {"empirical": None,
                      "shrunk": ShrunkCovariance(assume_centered=True),
                      "ledoit": LedoitWolf(assume_centered=True)}
    conditions = set(df["condition"])
    if estimator_names is None:
        estimator_names = ["empirical", "shrunk", "ledoit"]
    if kinds is None:
        kinds = ["correlation", "precision", "tangent", "partial correlation"]

    stat_data = []
    for kind in kinds:
        for estimator_name in estimator_names:
            estimators = {'kind': kind,
                          'cov_estimator': cov_estimator[estimator_name]}
            stat_df = statistical_test(df, conditions, p_correction="fdr",
                                       estimators=estimators)
            stat_data.append((stat_df, estimator_name, kind))
    return stat_data


def visualize_statistical_df(stat_df, estimator_name, kind, fig_dir,
                             comparisons=None, p_th=0.05, overwrite=False):
    """Plots the results from a statistical data frame.

    Parameters
    ==========
    stat_df: class 'pandas.core.frame.DataFrame'
        Data frame with statistical results.
    estimator_name: str
        Name of the estimator.
    kind: str
        Connectivity measure.
    fig_dir: str
        Directory for saving the figures.
    comparisons: set of tuple, optional
         Pairs of compared conditions to plot. If None, all included
         comparisons are plotted.
    p_th: float, optional, default to 0.05
        Significance threshold.
    overwrite: bool, optional, default to False
        If True, overwrites existing figures files.
    """
    all_comparisons = stat_df["comparison"]
    if comparisons is None:
        comparisons = all_comparisons

    for n, comp in enumerate(comparisons):
        if comp not in list(all_comparisons):
            continue
        b = stat_df["baseline"][n]
        f = stat_df["follow up"][n]
        matrices = [vec_to_sym(stat_df["mean" + name][n] * \
            (stat_df["pval" + name][n] < p_th), isometry=False) for
            name in [" follow up", " baseline", ""]]
        plt.figure(figsize=(10, 7))
        names = [f, b, comp]
        for i, matr in enumerate(matrices):
            plt.subplot(1, 3, 1 + i)
            plot_matrix(matr, plt.gca(), title=names[i], ticks=[],
                        tickLabels=[], symmetric_cb=True,
                        ylabel="significative mean")
            plt.draw()
        kind_dir = kind.replace(" ", "_")
        if not os.path.isdir(fig_dir):
            os.mkdir(fig_dir)

        fig_title = comp.replace(" ", "_") + "_" + estimator_name + "_" + \
            kind_dir
        file_path = os.path.join(fig_dir, fig_title + ".pdf")
        if not os.path.isfile(file_path) or overwrite:
            plt.savefig(file_path)
            os.system("pdfcrop %s %s" % (file_path, file_path))
    #    plt.show()


if __name__ == "__main__":
    data_set = "ds107"
    root_dir = os.path.join("/media/Elements/workspace/brainpedia/preproc",
                            "salma")
    cache_dir = os.path.join("/media/Elements/workspace/parietal_retreat/",
                             "salma", "visual_AN_DMN_noHv")
    save_dir = os.path.join(cache_dir, "figures", "uncorrected")
    n_jobs = 1
    maps_img = os.path.join("/volatile/new/salma/conn/genereated_rois",
                            "Visual_AN_DMN_labels_ordered.nii")
    df = get_data(root_dir=root_dir, cache_dir=cache_dir, data_set=data_set,
                  n_jobs=n_jobs, maps_img=maps_img)
    conditions = _get_conditions(root_dir=root_dir, data_set=data_set)
    cov_estimator = {"empirical": None,
                  "shrunk": ShrunkCovariance(assume_centered=True),
                  "ledoit": LedoitWolf(assume_centered=True)}
    if True:
        for kind in ["correlation"]:
            t_df = statistical_test(df, conditions,
                                    estimators={'kind': kind,
                                                'cov_estimator': cov_estimator["empirical"]},
                                    p_correction=None)
            labels = ["vPVC", "dPVC", "vIPS", "pIPS", "MT", "FEF",
                      "RTPJ", "RDLPFC", "AG", "SFG", "PCC", "MPFC", "FP"]
            plot_results(t_df, save_dir=save_dir, p_th=.05,
                         estim_title="empirical_" + kind,
                         labels=labels)