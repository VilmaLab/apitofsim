# Getting started with apitofsim

This is the documentation for apitofsim, a simulation of cluster fragmentation in an Atmospheric Pressure interface Time of Flight Mass Spectrometer (APi-ToF MS). This page will help you get started with installation and usage. For more information about the package itself, please refer to the [Background](background.md) page.

## Installation

This package can be installed using packages from Conda or pip (wheel) or from source

### Installation using pip

Visit https://github.com/VilmaLab/apitofsim/releases and find the appropriate release ending .whl for your platform, right click and copy the link address. Then run the following command in your terminal, replacing `<link>` with the copied link:

    pip install <link>

Or using uv:

    uv tool install <link>

### Installation using Conda

It is recommend to install this package using Conda.
First download [miniforge](https://github.com/conda-forge/miniforge) and then run:

    conda install -c https://prefix.dev/vilma apitofsim

### Build from source

It is also possible to [build from source](development.md).

## Usage

There are two main entry points to running the simulation: the command line tools and a Python library.
Basic functionality is available in both, but the Python API allows for more customization.

### Using apitofsim from the command line

First, refer to the [guide](guide.md) which walks through using all major features of the package through the command line. Next, you can refer to the reference for [the command line tools](cli.md) and for [the configuration files](configuration.md).

### Using apitofsim from Python

See [the reference documentation](python.md).
