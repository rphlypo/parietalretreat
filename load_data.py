import os.path
import string


from nipy.modalities.fmri import hrf, utils
from collections import Iterable


hrf_func = utils.lambdify_t(hrf.glover(utils.T))


def _get_paths(data_dict):
    path = dict()
    if not isinstance(data_dict, dict):
        raise ValueError("'data_dict' must be a dictionary")
    if not "func" in data_dict.keys():
        raise LookupError("at least 'func' must appear as a key in data_dict"
                          " to allow localisation of the bold images")

    _check_args(data_dict, data_type="func")
    path["func"] = os.path.normpath(data_dict["func"].format(**data_dict))

    # Confounds
    for data_type in {"conf", "anat", "onsets"} & set(data_dict.keys()):
        _check_args(data_dict, data_type=data_type)
        path[data_type] = os.path.normpath(
            data_dict[data_type].format(**data_dict))

    return path


def _check_args(data_dict, data_type="func"):
    str_out = ""
    str_format = string.Formatter()
    print list(str_format.parse(data_dict[data_type]))
    needed_keys = zip(*list(str_format.parse(data_dict[data_type])))[1]
    if needed_keys[-1] is None:
        needed_keys = needed_keys[:-1]
    for key in needed_keys:
        str_out += format(key) if str_out == "" else ", " + format(key)
    if not all([key in data_dict.keys() for key in needed_keys]):
        raise LookupError("format key is missing in argument list: "
                          "necessary format keys are " + str_out)
    return True


def dict_2_paths(arg_list):
    """
    Given a (list of) dictionary(ies), evaluate the path strings

    parameters:
    ----------
    arg_list: either an iterable of dict objects or a single dict object
        make sure the "func" key appears in the dictionaries which is an
        unformatted string, using the arguments appearing in the dictionary
        allowed paths are "func", "anat", "onsets", "confds"

    output:
    ------
    paths: a (list of) dictionary(ies)
        paths that are formatted using the arguments from the dictionaries
    """
    if isinstance(arg_list, Iterable) and not isinstance(arg_list, dict):
        return [dict_2_paths(arg) for arg in arg_list]
    else:
        return _get_paths(arg_list)
