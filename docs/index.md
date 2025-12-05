# Documentation for apitofsim

This is the documentation for apitofsim, a simulation of cluster fragmentation in an Atmospheric Pressure interface Time of Flight Mass Spectrometer (APi-ToF MS).

## Installation

It is recommend to install this package using Conda.
Users on Windows should use WSL.
First download [miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#unix-like-platforms-macos-linux--wsl) and then run:

    conda install -c https://prefix.dev/vilma apitofsim

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

## How does it work?

The simulation runs a number of iterations with each one considering a single instance of a cluster travelling through and APi-TOF MS.
The main simulation loop considers the distributions of the time until the next collision between the cluster and a gas molecules, the speed/angle of that collision, and the time until the cluster fragments. The main quantity of interest is the probability the cluster survives to reach the detector without fragmenting.

### Publications describing the simulation

The main principles of the simulation are described across a number of publications.

Zapadinsky et al. (2019)[^1] describe the simplest version of the simulation, in which only a single pressure and electric field are considered.
Further information on the equations are given in the supporting information[^2].
The main parts described are the overall scheme of simulation.
The actual code used for this publication was written in Matlab and is not publicly available.

Zanca et al. (2020)[^3] describe a version of the simulation expanding the above to consider five zones.
Zone I being the first chamber, II the skimmer, and III-V the second chamber, before, during and after the quadrupole respectively.
Simulation of the skimmer and quadrupole is described.
The code used in this publication is an earlier version of the code in this repository, but is not publicly available.

[^1]:
    Zapadinsky, E., Passananti, M., Myllys, N., Kurtén, T., & Vehkamäki, H. (2019).
    Modeling on Fragmentation of Clusters inside a Mass Spectrometer.
    *The Journal of Physical Chemistry. A*, 123, 611 - 624.
    [[web]](https://pubs.acs.org/doi/10.1021/acs.jpca.8b10744) [[pdf]](https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.8b10744?ref=article_openPDF) [[doi]](https://doi.org/10.1021/acs.jpca.8b10744)
[^2]:
    Zapadinsky, E., Passananti, M., Myllys, N., Kurtén, T., & Vehkamäki, H. (2019).
    Supporting Information to "Modelling on Fragmentation of Clusters Inside a Mass Spectrometer"
    [[pdf]](https://pubs.acs.org/doi/abs/10.1021/acs.jpca.8b10744)
[^3]:
    Zanca, T., Kubečka, J., Zapadinsky, E., Passananti, M., Kurtén, T., & Vehkamäki, H. (2020).
    Highly oxygenated organic molecule cluster decomposition in atmospheric pressure interface time-of-flight mass spectrometers.
    *Atmospheric Measurement Techniques*, 13, 3581-3593.
    [[web]](https://amt.copernicus.org/articles/13/3581/2020/) [[pdf]](https://amt.copernicus.org/articles/13/3581/2020/amt-13-3581-2020.pdf) [[doi]](https://doi.org/10.5194/amt-13-3581-2020)

## Publications using the simulation

These publication make use (previous versions of) this simulation.

**TODO: Complete this section.**

## Building the sources

This section describes how to build the sources from scratch.

### Using Meson to compile the Python extension

[meson-python](https://mesonbuild.com/meson-python/) is used to build the Python extension.

You need disable build isolation to install this as a development/editable package which will recompile at import:

Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv pip install --no-build-isolation -e .
```

During development you can also run the following to get compiler errors on import, and add debug symbols:

```bash
uv pip install --no-build-isolation --config-settings=editable-verbose=true -Csetup-args="-Dbuildtype=debugoptimized" -Cbuild-dir=pydebugbuild -e .
```

### Using Meson to compile the executables

```bash
meson setup --buildtype release build
meson compile -C build
```

The binaries are then in `build/src`. You add this directory to your PATH or symlink to them.

This will also create a compilation database that `clangd` can use.

There are debug and sanitize Meson "native build configuration" files using clang in `meson/clangdebug.ini` and `meson/clangsan.ini` respectively.
On Linux, you may need to install `libc++` (from LLVM rather than GNU) e.g. `apt install 'libc++1' 'libc++-dev'`.
For example:

```bash
meson setup --native-file meson/clangdebug.ini debugbuild
```
