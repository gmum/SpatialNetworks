def get(module, args):
    if args.input.lower() == "sequential":
        return getattr(module, "SingleOutput")
    return getattr(module, "MultipleOutputs")
