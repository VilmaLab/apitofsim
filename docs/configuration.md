# Configuration

This page acts as a reference for configuration files.
Note that much of the same structure is used [by the Python library](python.md).

Here's an example of a full configuration file used in the validation example (available in the repository at `examples/validation`).
You will probably want to skim this, but can come back after reading the sections below.

``` title="examples/validation/config.toml"
--8<-- "examples/validation/config.toml"
```

Note there are two main things configured in this file.
The first is how to ingest things, as given by one or more `[[pathways]]` sections.
The second is how to run the simulation, as given by the `[default_config]` section and one or more `[[config]]` sections.

## Ingestion configuration

See the documentation for the [apitofsim db prepare][apitofsim-db-prepare] command.

## Simulation parameters

For the simulation parameters, you can specify one or more `[[configs]]` sections, each with a `name` field and the values of all parameters.

You can put common parameters in the `default_config` section, so that each `[[configs]]` section inherits these as overridable defaults.

For parameters with units, a unit must be given.
Internally [pint](https://pint.readthedocs.io/en/stable/) is used to convert into the preferred units for the simulation, so you can specify any unit with the correct dimensions.
For scalars, units can be expressed either as a string (e.g. `1.0 meter`) or as an array of two elements, where the first element is the value and the second element is the unit (e.g. `[1.0, "meter"]`).
For arrays, the second for should be used e.g. `[[1.0, 2.0], "meter"]`.

For the following example, say we have a start a simulation parameters section like so:

```toml
[[configs]]
name = "myconfig"
```

### Top level scalars

| Name            | Typical unit | Description                          |
| --------------- | -------------|--------------------------------------|
| N_iter          | -            | Number of iterations to run initial grid search when performing root finding of the area-velocity equation when making histogrammed quasi-one-dimensional isentropic nozzle flow calculation for the skimmer |
| M_iter          | -            | Number of secant steps to perform when performing root finding of the area-velocity equation when making histogrammed quasi-one-dimensional isentropic nozzle flow calculation for the skimmer |
| tolerance       | -            | Tolerance when performing root finding of the area-velocity equation when making histogrammed quasi-one-dimensional isentropic nozzle flow calculation for the skimmer |
| resolution      | -            | Number of points to use when histogramming across length of the skimmer region (as specified in lengths) |
| N               | -            | Number of realizations to use in main particle simulation |
| bin_width       | kelvin       | Bin width for histogramming density of states and k_total |
| energy_max      | kelvin       | Maximum energy to consider when histogramming density of states |
| energy_max_rate | kelvin       | Maximum energy to consider when histogramming k_total |
| alpha_factor    | halfturn     | Angle of skimmer |
| T               | kelvin       | Temperature inside the mass spectrometer |
| dc              | meter        | Radius at smallest cross section of skimmer |

### Top level arrays

<figure markdown="span">
  ![An APi-ToF mass spectrometer with lengths](img/lengths.png){ width="300" }

<figcaption>Figure showing how the lengths and pressures correspond to different sections of APi-ToF mass spectrometer</figcaption>
</figure>

<figure markdown="span">
  ![An APi-ToF mass spectrometer with voltages](img/voltages.png){ width="300" }

<figcaption>Figure showing where the voltages are applied on an APi-ToF mass spectrometer</figcaption>
</figure>

| Name            | Typical unit | Description                          |
| --------------- | -------------|--------------------------------------|
| pressures       | pascal       | A two element array of the pressures in the first and second chamber, respectively [$P_1$, $P_2$] |
| ... $P_1$       | meter        | Pressure in 1st chamber |
| ... $P_2$       | meter        | Pressure in 2nd chamber |
| lengths         | meter        | A five element array with the important dimensions of the instrument [$L_0$, $L_1$, $L_2$, $L_3$, $L_{sk}$] |
| ... $L_0$       | meter        | Length of 1st chamber |
| ... $L_1$       | meter        | Length between skimmer and front quadrupole |
| ... $L_2$       | meter        | Length between front quadrupole and back quadrupole |
| ... $L_3$       | meter        | Length between back quadrupole and 2nd skimmer |
| ... $L_{sk}$    | meter        | Length of skimmer |
| voltages        | volt         | A five element array with the voltages applied to the instrument [$V_0$, $V_1$, $V_2$, $V_3$, $V_4$] |
| ... $V_0$       | volt         | Voltage at then end of the quadrupole in first chamber (not currently simulated) |
| ... $V_1$       | volt         | Voltage between first and second chambers |
| ... $V_2$       | volt         | Voltage at beginning of quadrupole in second chamber |
| ... $V_3$       | volt         | Voltage at end of quadrupole in second chamber |
| ... $V_4$       | volt         | Voltage at end of second chamber |

### The gas section

For example:

```toml
[[configs.gas]]
radius = "1.84e-10 meter"
mass = "4.65e-26 kilogram"
= 1.4
```

| Name            | Typical unit | Description                          |
| --------------- | -------------|--------------------------------------|
| radius          | meter        | The radius of the gas particle |
| mass            | kilogram     | The mass of the gas particle |
| adiabatic_index | -            | The adiabatic index of the gas |

### The quadrupole section

This section is optional. If not specified, the simulation will be run without a quadrupole.

For example:

```toml
[configs.quadrupole]
dc_field = "0.0 volt"
ac_field = "200.0 volt"
radiofrequency = "1.3e6 Hz"
r_quadrupole = "6.0e-3 meter"
```

| Name            | Typical unit | Description                          |
| --------------- | -------------|--------------------------------------|
| dc_field        | volt         | DC field at the quadrupole |
| ac_field        | ac_field     | AC field at the quadrupole |
| radiofrequency  | hertz        | Radio frequency |
| r_quadrupole    | meter        | Half-distance between quadrupole rods |

## Legacy configuration (.in file)

This format is order dependent and its usage is not recommended.

An example is reproduced here for reference:

``` title="config.in"
--8<-- "inputs/example/config.in"
```

