# Adding to codebase

## New functionality

If you wish to add another functionality, which doesn't fit into current
`main.py`'s subcommand, you should do the following:

1. Create new subparser within `/src/inputs/subparsers.py` (check examples there)
2. Register subparse in main parser within `/src/inputs/parser.py`
3. Create new package in `/src/options/` named just like the new subparser options you added
4. Inside `/src/options/__init__.py` create function `run` taking one argument `args`
(which will be the general and your subparser specific use provided options).
5. Do what your new options requires.

## Adding new functionality to `experiment` wrapper (optional)

Using `/src/experiment` one can run all experiments and cache them using [DVC](https://dvc.org/).
Using this wrapper, if certain stage was already performed, it will not be recalculated.
To register your new functionality you should do the following:

1. Create new function performing your operation and displaying current stage.

    ```shell
    <new_subcommand>() {
      green_message "<NEW_SUBCOMMAND>"
      dvc run [RUN_OPTIONS] python main.py [GENERAL_OPTIONS] <new_subcommand> [SUBCOMMAND_OPTIONS]
      green_message "<NEW_SUBCOMMAND> SUCCESSFUL"
    }
    ```

    `[RUN_OPTIONS]` are `dvc` specific like `-f` (name of the file for reproduction)
    or `-d` (file and stage dependencies).
    See section `FUNCTIONS` and `train` for example.

2. Define folders where experiment created data will reside. Things like saved plots,
saved models and others. Those should be contained within created folder inside `$DATA_PATH`.
See section `DATA AND EXPERIMENTS FOLDER` for examples.


3. Define file where DVC will save necessary steps for reproduction. Those should follow
the scheme below:

    ```shell
    <NEW_SUBCOMMAND>_FILE="$EXPERIMENTS_PATH/<new_subcommand>_$EXPERIMENT_SPECIFIC.dvc"
    ```

    Provided other variables are already defined. See `DVC FILES FOR REPRODUCTION`.

4.  Run your new functionality with appropriate dependencies and artifacts created
in appropriate folders. See section `RUN SUBCOMMAND AND CREATE ARTIFACTS`
