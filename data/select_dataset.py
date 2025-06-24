

import importlib

def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type']
    try:
        D = getattr(importlib.import_module('data.dataset_{}'.format(dataset_type), package=None), 'Dataset{}'.format(dataset_type))
    except Exception:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
