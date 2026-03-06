# Getting started with apitofsim

This is the documentation for apitofsim, a simulation of cluster fragmentation in an Atmospheric Pressure interface Time of Flight Mass Spectrometer (APi-ToF MS). This page will help you get started with installation and usage. For more information about the package itself, please refer to the [Background](background.md) page.

## Installation

It is recommend to install this package using Conda.
Users on Windows should use WSL.
First download [miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl) and then run:

    conda install -c https://prefix.dev/vilma apitofsim

It is also possible to [build from source](development.md).

## Usage

There are two main entry points to running the simulation: the command line tools and a Python library.
Basic functionality is available in both, but the Python API allows for more customization.

### Using apitofsim from the command line

First, refer to the [guide](guide.md) which walks through using all major features of the package through the command line. Next, you can refer to the reference for [the command line tools](cli.md) and for [the configuration files](configuration.md).

### Using apitofsim from Python

See [the reference documentation](python.md).
