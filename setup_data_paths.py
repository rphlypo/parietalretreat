import glob
import os.path
from pandas import DataFrame
import pandas


def get_all_paths(data_set=None, root_dir="/"):
    list_ = list()
    head, tail_ = os.path.split(root_dir)
    counter = 0
    while tail_:
        head, tail_ = os.path.split(head)
        counter += 1
    if data_set is None:
        df_ = list()
        data_sets = {"hcp", "henson2010faces", "ds105", "ds107"}
        for ds in data_sets:
            df_.append(get_all_paths(data_set=ds, root_dir=root_dir))
        df = pandas.concat(df_, keys=data_sets)
    elif data_set.startswith("ds") or data_set == "henson2010faces":
        base_path = os.path.join(root_dir,
                                 "storage/workspace/brainpedia/preproc/",
                                 data_set)
        with open(os.path.join(base_path, "scan_key.txt")) as file_:
            TR = file_.readline()[3:-1]
        for fun_path in glob.iglob(os.path.join(base_path,
                                                "sub*/model/model*/"
                                                "BOLD/task*/bold.nii.gz")):
            head, tail_ = os.path.split(fun_path)
            tail = [tail_]
            while tail_:
                head, tail_ = os.path.split(head)
                tail.append(tail_)
            tail.reverse()
            subj_id = tail[6 + counter][-3:]
            model = tail[8 + counter][-3:]
            task, run = tail[10 + counter].split("_")

            tmp_base = os.path.split(os.path.split(fun_path)[0])[0]

            anat = os.path.join(tmp_base,
                                "anatomy",
                                "highres{}.nii.gz".format(model[-3:]))

            onsets = glob.glob(os.path.join(tmp_base, "onsets",
                                            "task{}_run{}".format(task, run),
                                            "cond*.txt"))

            confds = os.path.join(os.path.split(fun_path)[0], "motion.txt")
            list_.append({"subj_id": subj_id,
                          "model": model,
                          "task": task[-3:],
                          "run": run[-3:],
                          "func": fun_path,
                          "anat": anat,
                          "confds": confds,
                          "TR": TR})
            if onsets:
                list_[-1]["onsets"] = onsets

        df = DataFrame(list_)
    elif data_set == "hcp":
        base_path = os.path.join(root_dir, "storage/data/HCP/Q2/")
        for fun_path in glob.iglob(os.path.join(base_path,
                                                "*/MNINonLinear/Results/",
                                                "*/*.nii.gz")):

            head, tail = os.path.split(fun_path)
            if head[-2:] not in ["LR", "RL"]:
                continue
            tail = [tail]
            while head != "/":
                head, t = os.path.split(head)
                tail.append(t)
            if tail[0][:-7] != tail[1]:
                continue
            tail.reverse()
            subj_id = tail[4 + counter]
            task = tail[7 + counter][6:-3]
            if tail[7 + counter].startswith("rfMRI"):
                run = task[-1]
                task = task[:-1]
            mode = tail[7 + counter][-2:]

            anat = os.path.join(base_path, subj_id, "MNINonLinear/T1w.nii.gz")

            confds = os.path.join(os.path.split(fun_path)[0],
                                  "Movement_Regressors.txt")
            list_.append({"subj_id": subj_id,
                          "task": task,
                          "mode": mode,
                          "func": fun_path,
                          "anat": anat,
                          "confds": confds,
                          "TR": 0.72})
            if tail[8 + counter].startswith("rfMRI"):
                list_[-1]["run"] = run
            else:
                onsets = [onset
                          for onset in glob.glob(os.path.join(
                              os.path.split(fun_path)[0], "EVs/*.txt"))
                          if os.path.split(onset)[1][0] != "S"]
                list_[-1]["onsets"] = onsets
        df = DataFrame(list_)
    return df


if __name__ == "__main__":
    from nilearn.input_data import MultiNiftiMasker, NiftiMapsMasker
    from joblib import Memory, Parallel, delayed
    from sklearn.base import clone
    import nibabel

    mem = Memory(cachedir="/storage/workspace/rphlypo/retreat/dump/")
    print "Loading all paths and variables into memory"
    df = get_all_paths()
    target_affine_ = nibabel.load(df["func"][0]).get_affine()
    target_shape_ = nibabel.load(df["func"][0]).shape[:-1]
    print "preparing and running MultiNiftiMasker"
    mnm = MultiNiftiMasker(mask_strategy="epi", memory=mem, n_jobs=10,
                           verbose=10, target_affine=target_affine_,
                           target_shape=target_shape_)
    mask_img = mnm.fit(list(df["func"])).mask_img_
    print "preparing and running NiftiMapsMasker"
    nmm = NiftiMapsMasker(
        maps_img=os.path.join("/usr/share/fsl/data/atlases/HarvardOxford/",
                              "HarvardOxford-cortl-prob-2mm.nii.gz"),
        mask_img=mask_img, detrend=True, smoothing_fwhm=5, standardize=True,
        low_pass=None, high_pass=None, memory=mem, verbose=10)
    region_ts = Parallel(n_jobs=25)(delayed(clone(nmm).fit_transform)
                                        (niimg, n_hv_confounds=5)
                                    for niimg in list(df["func"]))
