# import nilearn
# import sklearn
import setup_data_paths
import numpy as np
import joblib
import os.path
from pandas import DataFrame


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
        print names
        data = list()
        for ix_ in range(len(group)):
            onset_file, TR, region_ix =\
                [group.iloc[ix_][k]
                 for k in ["cond_onsets", "TR", "region_ix"]]
            data.append(get_samples(region_signals[ix_],
                                    onset_file, TR))
        arr = np.vstack(data)
        df_list.append({"condition": names[0],
                        "subj_id": names[1],
                        "region_signals": arr})
    return DataFrame(df_list)
    #conditions = get_conditions(data_set)
    #region_condition_data = [region_signals[]]


def get_samples(signals, onset_file, TR):
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

#
#        #plt.step(X, y)
#    #plt.ylim((-.1, 4.1))
#    #plt.show()
#    # df[region_signals] contain the region signals
#    # df[onsets] contain the different onsets
