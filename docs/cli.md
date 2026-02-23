# Using the command line tools

If you have installed via Conda, and activated the relevant environment, the command line tool `apitofsim` should be installed into your path.
If you are using `uv` you may need to run `uv run apitofsim` instead.

::: mkdocs-click
    :module: apitofsim.cli
    :command: cli
    :prog_name: apitofsim
    :depth: 1
    :list_subcommands: True

## The legacy tools

These tools are not recommended for new users, and kept available to support existing workflows.

If you have installed via Conda, and activated the relevant environment, the legacy command line tools `apitofsim-skimmer`, `apitofsim-densityandrate` and `apitofsim-mass-spec` should be installed and in your path.
If you have compiled the sources yourself, you will need to add build/src to your path for the following example to work.
You can run the included example pathway like so:

```bash
apitofsim-skimmer < inputs/example/config.in
apitofsim-densityandrate < inputs/example/config.in
apitofsim-mass-spec < inputs/example/config.in
```

Outputs are generated in `work/out` directory.
