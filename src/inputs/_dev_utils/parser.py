def validate_spatial_arguments(args):
    if args.command == "train" and args.where is not None:
        if args.proximity is None or args.transport is None or args.norm is None:
            raise ValueError(
                "Spatial layers were specified. Values of alpha, beta and norm have to be specified."
            )


def validate_spatial_locations(args):
    if args.command == "train" and args.where is not None:
        if max(args.where) > len(args.layers):
            raise ValueError(
                "One (or more) of indices --where module should be Spatial is greater "
                + "than total count of --layers specified. "
            )

        if any(value < 0 for value in args.where):
            raise ValueError(
                "Only indices zero or greater can be specified in --where flag."
            )


def process_arguments(args):
    validate_spatial_arguments(args)
    validate_spatial_locations(args)
    return args
