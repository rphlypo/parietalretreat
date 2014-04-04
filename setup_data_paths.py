import load_data
import glob
import re
import os.path
import itertools
reload(load_data)


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


def get_hcp_task_files(subject_list, session_list, scan_mode_list,
                       task_list):
    """ obtain the hcp file pointers
    """
    keys = ("func", "anat", "onsets", "conf",
            "subj_id", "session", "scan_mode", "task")
    values = ("/storage/data/HCP/Q2/{subj_id}/MNINonLinear/Results/" +
              "tfMRI_{task}_{scan_mode}/tfMRI_{task}_{scan_mode}.nii.gz",
              "/storage/data/HCP/Q2/{subj_id}/MNINonLinear/T1w.nii.gz",
              "/storage/data/HCP/Q2/{subj_id}/MNINonLinear/Results/" +
              "tfMRI_{task}_{scan_mode}/EVs/",
              "/storage/data/HCP/Q2/{subj_id}/MNINonLinear/Results/" +
              "tfMRI_{task}_{scan_mode}/Movement_Regressors.txt")

    list_of_dicts = [dict(zip(*[keys, values + spec_values]))
                     for spec_values in
                     itertools.product(subject_list,
                                       session_list,
                                       scan_mode_list,
                                       task_list)]
    return load_data.dict_2_paths(list_of_dicts)


def get_ds_task_files(subject_list, session_list, task_list, run_list,
                      model_list, cond_list):
    """
    """
    keys = ("func", "anat", "onsets", "conf",
            "subj_id", "session", "task")
    values = ("/storage/workspace/brainpedia/preproc/{study}/" +
              "sub{subj_id}/model/model{model_id}/BOLD/task{task_id}_" +
              "run{run_id}/bold.nii.gz",
              "/storage/workspace/brainpedia/preproc/{study}/" +
              "sub{subj_id}/model/model{model_id}/anatomy/" +
              "highres{model_id}.nii.gz",
              "/storage/workspace/brainpedia/preproc/{study}/" +
              "sub{subj_id}/model/model{model_id}/onsets/" +
              "task{task_id}_run{run_id}/cond{cond_id}.txt",
              "/storage/workspace/brainpedia/preproc/{study}/" +
              "sub{subj_id}/model/model{model_id}/BOLD/task{task_id}_" +
              "run{run_id}/motion.txt")

    list_of_dicts = [dict(zip(*[keys, values + spec_values]))
                     for spec_values in
                     itertools.product(subject_list,
                                       session_list,
                                       task_list,
                                       run_list,
                                       model_list,
                                       cond_list)]
    return load_data.dict_2_paths(list_of_dicts)
