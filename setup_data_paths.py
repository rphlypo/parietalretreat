import load_data
import glob
import re
import os.path


def get_subject_ids(data_set):
    """
    for a given dataset, get the different subjects as a list

    this allows for setting up the lists of the data paths
    but is also convenient for sampling of train-test splitting based
    on the subject list

    arguments:
    ---------
    data_set: string
        at this moment the only datasets that are supported are
        "hcp", "henson2010faces", or openfmri datasets "ds???"

    returns:
    -------
    subj_ids: list of strings with subject ids in the dataset
    """
    if data_set == "hcp":
        # subjects from HCP
        base_dir = "/home/storage/data/HCP/Q2"
        subj_str = "[0-9]" * 6
    elif data_set.lower()[:2] == "ds" or data_set == "henson2010faces":
        # subject from any openfmri data set
        base_dir = os.path.join("/home/storage/workspace/brainpedia/preproc",
                                data_set)
        subj_str = "sub" + "[0-9]" * 3

    return _subject_ids(base_dir, subj_str)


def _subject_ids(base_dir, subj_str):
    regexp = re.compile(os.path.join(base_dir, format(subj_str)))
    subject_paths = glob.glob(os.path.join(base_dir, "*"))
    subject_paths = [p for p in subject_paths if regexp.match(p)]
    return [os.path.basename(subj_path) for subj_path in subject_paths]
