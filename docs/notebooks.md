# Running the notebooks

There are notebooks in the [`examples/notebooks`](https://github.com/VilmaLab/apitofsim/tree/main/examples/notebooks) directory.

In order to run them, first download them from GitHub, and then install [uv](https://docs.astral.sh/uv/getting-started/installation/) according to the linked instructions and then install marimo like so:

```
uv tool install marimo
```

You can then run a notebook using:

```
marimo edit --sandbox rejection-sampler-design.py
```
