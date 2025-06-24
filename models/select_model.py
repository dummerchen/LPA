

def define_Model(opt, **kwargs):
    model = opt['models']

    if model == 'plain':
        from models.model_plain import ModelPlain as M

    else:
        try:
            import importlib
            M = getattr(importlib.import_module('models.model_' + model.lower(), package=None), 'Model'+model)
        except Exception as e:
            raise NotImplementedError(e, 'Model [{:s}] is not defined.'.format(model))

    m = M(opt, **kwargs)
    return m
