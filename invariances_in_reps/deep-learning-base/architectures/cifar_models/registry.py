model_names, model_fns = [], {}

def register_model_name(fn):
    model_names.append(fn.__name__)
    model_fns[fn.__name__] = fn

def create_model_fn(model_name):
    return model_fns[model_name]
