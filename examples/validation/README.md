To run this workflow, you must first obtain the data for [^1], which is currently available by request only.

Step 1: Modify the file `config.toml` so that the 3 paths starting `/path/to` point to the location of the data on your system.

Step 2: Prepare the dataset:


```bash
$ apitofsim db prepare create config.toml validation.duckdb
```

Step 3: Run the simulations. You can either run the following command locally but you might like to use a cluster since it will currently take around 10 hours on a 64 core machine. See `run.turso.uh.slurm` for a SLURM example or else run:

```bash
$ apitofsim db run validation.duckdb
$ apitofsim db run --pathway-at-a-time validation.duckdb
```

Now you can run the notebook to compare survival probabilities and spectrograms according to different methods as well the results from the paper.

Note you will need to obtain `original_results_pathways.csv`.

```bash
uv run marimo plots.mo.py
```

This dataset may be published as open data at some point in the future.

[1]: Alfaouri, D., Passananti, M., Zanca, T., Ahonen, L.R., Kangasluoma, J., Kubečka, J., Myllys, N., & Vehkamäki, H. (2022). A study on the fragmentation of sulfuric acid and dimethylamine clusters inside an atmospheric pressure interface time-of-flight mass spectrometer. Atmospheric Measurement Techniques. [[doi]](https://doi.org/10.5194/amt-15-11-2022)
