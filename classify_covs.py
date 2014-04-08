# import nilearn
# import sklearn
import setup_data_paths
import numpy as np
import joblib
import os.path
from pandas import DataFrame
import confound
import scipy.linalg


def load_data(root_dir="/",
              data_set="ds107",
              dump_file="storage/workspace/parietal_retreat/" +
                       "covariance_learn/dump/results.pkl"):
    df = setup_data_paths.get_all_paths(root_dir=root_dir, data_set="ds107")
    region_signals = joblib.load(os.path.join(root_dir,
                                              dump_file))
    return df, region_signals


def get_conditions(root_dir, data_set):
    data_set_dir = os.path.join(root_dir,
                                "/storage/workspace/brainpedia/preproc",
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


def get_region_signals(df, region_signals, data_set="ds107"):
    df_ = df.groupby(["condition", "subj_id"])
    df_list = list()
    for names, group in df_:
        data = list()
        for ix_ in range(len(group)):
            onset_file, TR, region_ix, confd_file =\
                [group.iloc[ix_][k]
                 for k in ["cond_onsets", "TR", "region_ix", "confds"]]
            confds = confound.compute_mvt_confounds(confd_file)[0]
            signals = _regress(region_signals[ix_], confds)
            data.append(get_samples(signals, onset_file, TR))
        arr = np.vstack(data)
        df_list.append({"condition": names[0],
                        "subj_id": names[1],
                        "region_signals": arr})
    return DataFrame(df_list)


def get_samples(signals, onset_file, TR, confds):
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


def get_symm_psd_mx(df, CovEst):
    covs = list()
    for ix_ in range(len(df)):
        time_series = df.iloc[ix_]["region_signals"]
        cov_est = CovEst(assume_centered=True)
        covs.append(cov_est.fit(time_series).covariance_)
    df["covs"] = covs
    return df


def _regress(X, y):
    print X.shape, y.shape
    Q, _ = scipy.linalg.qr(y, mode="economic")
    return X - y.dot(np.linalg.pinv(y.T.dot(y))).dot(y.T.dot(X))


if __name__ == "__main__":
    df, region_signals = load_data(root_dir="/media/Elements/volatile/new/salma",
                                   data_set="ds107")

    df2 = get_region_signals(df, region_signals)
    groups = df2.groupby("condition")
    for condition, group in groups:
        print "condition = {}".format(condition)
        for ix_ in range(len(group)):
            print "\tsubj_id = {}, shape = {}".format(
                group.iloc[ix_]["subj_id"],
                group.iloc[ix_]["region_signals"].shape)
