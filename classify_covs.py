import sys
import copy
import os.path

import numpy as np
import joblib
from pandas import DataFrame

import scipy.linalg
import scipy.stats.mstats

from connectivity import CovEmbedding
import setup_data_paths
reload(setup_data_paths)
import confound


def get_data(root_dir="/",
             data_set="ds107",
             dump_file="/home/storage/workspace/parietal_retreat/" +
             "covariance_learn/dump/results.pkl"):
    df, region_signals = _load_data(root_dir=root_dir,
                                    data_set=data_set,
                                    dump_file=dump_file)
    return _get_region_signals(df, region_signals, data_set=data_set)


def _load_data(root_dir="/",
               data_set="ds107",
               dump_file="/home/storage/workspace/parietal_retreat/" +
               "covariance_learn/dump/results.pkl"):
    df = setup_data_paths.get_all_paths(root_dir=root_dir, data_set="ds107")
    region_signals = joblib.load(os.path.join(root_dir, dump_file))
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
    Z[corr_is_one] = np.nan
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
    X: array
        transformed data, same shape as Y
    """
    if kind in ['correlation', 'partial correlation']:
        Y = corr_to_Z(X)
    else:
        Y = X

    return Y


def statistical_test(estimators={'kind': 'tangent',
                                 'cov_estimator': None},
                     root_dir="/"):
    df = get_data(root_dir=root_dir)
    grouped = df.groupby(["condition", "subj_id"])
    conditions = _get_conditions(root_dir=root_dir)
    dict_list = list()
    entries = ("comparison", "tstat", "pval")
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
            Y = var_stabilize(X, estimators['kind'])
            t_stat, p = scipy.stats.mstats.ttest_rel(Y[::2, ...],
                                                     Y[1::2, ...],
                                                     axis=0)
            print "{} vs. {}: t_stat = {}, p-val = {}".format(
                condition1, condition2, t_stat, p)
            dict_list.append(
                dict(zip(*[entries,
                           ("{} vs. {}".format(condition1, condition2),
                            t_stat, p)])))
    return DataFrame(dict_list, columns=entries)

if __name__ == "__main__":
    t_test = statistical_test(root_dir="/home")
