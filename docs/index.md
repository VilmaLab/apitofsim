# Getting started with apitofsim

This is the documentation for apitofsim, a simulation of cluster fragmentation in an Atmospheric Pressure interface Time of Flight Mass Spectrometer (APi-ToF MS). This page will help you get started with installation and usage. For more information about the package itself, please refer to the [Background](background.md) page.

## Installation

It is recommend to install this package using Conda.
Users on Windows should use WSL.
First download [miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl) and then run:

    conda install -c https://prefix.dev/vilma apitofsim

It is also possible to [build from source](development.md).

## Usage

There are two main entry points to running the simulation: the Python API and the command line tools.
The Python API is recommended for new users, and new functionality may only be available there.

### Python API

See [Using the Python API](api.md).

### Command line tools

If you have installed via Conda, and activated the relevant environment, the command line tools should be installed and in your path.
If you have compiled the sources yourself, you will need to add build/src to your path for the following example to work.
You can run the included example pathway like so:

```bash
apitofsim-skimmer < inputs/example/config.in
apitofsim-densityandrate < inputs/example/config.in
apitofsim-main < inputs/example/config.in
```

Outputs are generated in `work/out` directory.
