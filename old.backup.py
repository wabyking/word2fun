
def load_model(model, filename="pytorch.bin"):
    state_dict = torch.load(filename)
    print(filename)
    print(state_dict.keys())
    print(state_dict.__class__.__name__)
    exit()
    missing_keys, unexpected_keys, error_msgs = [], [], []
    prefix = ""
    metadata = getattr(state_dict, "_metadata", "None")
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys,
                                     error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    start_prefix = ""
    load(model, prefix=start_prefix)

    if len(missing_keys) > 0:
        print("weights of {} not initialized from pretrained model: {}".format(model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("weights of {} not used pretrained model: {}".format(model.__class__.__name__, unexpected_keys))
    if len(error_msgs) > 0:
        print("errors in loading state_dict  for  {}  :  \n{}".format(model.__class__.__name__, error_msgs))
    return model
