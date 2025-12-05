To run this workflow, you must first obtain the data for [^1], which is currently available by request only.

To run the workflow, create a file `config.json` pointing to the dataset:

```
{
  "path": "/path/to/data/100DMA-100SA/**/*.in",
  "cwd": ".",
  "backup_search": "/path/to/data/QCData"
}
```

Then import the data using the provided script:

```
uv run prepare_data.py config.json prepared_data
```

You are now ready to run the workflow:

```
uv run run.py prepared_data
```

Plotting code will be provided later. This dataset may be published as open data at some point in the future.

[1]: Alfaouri, D., Passananti, M., Zanca, T., Ahonen, L.R., Kangasluoma, J., Kubečka, J., Myllys, N., & Vehkamäki, H. (2022). A study on the fragmentation of sulfuric acid and dimethylamine clusters inside an atmospheric pressure interface time-of-flight mass spectrometer. Atmospheric Measurement Techniques. [[doi]](https://doi.org/10.5194/amt-15-11-2022)
