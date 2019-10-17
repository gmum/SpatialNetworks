def print_verbose(message, verbose: bool, *args, **kwargs):
    if verbose:
        print(message, *args, **kwargs)
