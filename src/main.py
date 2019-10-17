import warnings

with warnings.catch_warnings():
    # Tensorboard daily struggle
    warnings.simplefilter(action="ignore", category=FutureWarning)
    import importlib

    import inputs
    import options

if __name__ == "__main__":
    with warnings.catch_warnings():
        # Numpy daily struggle
        warnings.simplefilter(action="ignore", category=RuntimeWarning)
        # Source cannot be retrieved for inferred modules, to be fixed
        # So it's torchlayers (e.g. mine) struggle this time xD
        warnings.simplefilter(action="ignore", category=UserWarning)
        args = inputs.parser.get()
        module = importlib.import_module("options." + args.command)
        module.run(args)
