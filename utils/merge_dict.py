import collections

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict) and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            if k in dct.keys():
                if isinstance(dct[k], list):  # append lists
                    dct[k] += merge_dct[k]
                else:
                    dct[k] = merge_dct[k]
            else:
                dct[k] = merge_dct[k]




def run_test():
    output1 = {
        'name': 'arc',
        'color': 'yellow',
        'performance': {
            'filename': ['rec1', 'rec2'],
            'kappa': [0.45, 0.8]
        },
        'output_all': {
            'wake': {'tar': [0., 0., 1.0], 'pre': [0.1, 0.2, 0.3]},
            'sleep': {'tar': [1., 1., 0.], 'pre': [0.9, 0.8, 0.6]},
        }
    }

    output2 = {
        'name': 'arc',
        'color': 'yellow',
        'performance': {
            'filename': ['rec3', 'rec4'],
            'kappa': [0.5, 0.89]
        },
        'output_all': {
            'wake': {'tar': [1., 0., 1.0], 'pre': [0.12, 0.23, 0.34]},
            'sleep': {'tar': [0., 1., 0.], 'pre': [0.69, 0.58, 0.65]},
        }
    }

    dict_merge(output1, output2)

