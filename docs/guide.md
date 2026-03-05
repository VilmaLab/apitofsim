# Guide: from importing cluster data to spectrogram plots

This guide will take you through all the major features of the package by going through the process of importing cluster data, preparing parameters, and running the simulation through to using the simulation data to produce plots such as spectrograms.

## Importing cluster data

In the import or ingestion stage, you will prepare a database containing data about the clusters you want to simulate as well as which pathways to simulate.

Typically, you will start with an XYZ file and run either ORCA or Gaussian to generate the quantum chemistry.

The exact ORCA and Gaussian settings depend upon the material of the clusters being studied and the desired level of accuracy versus available computation time, and are somewhat beyond the scope of this guide.
Sometimes multiple analyses are run to estimate different properties of the cluster, for example one which is more accurate for energetics, and one which is more accurate for vibrational and rotational frequencies.
The following examples given only as an indication of settings which have been used previously:

=== "ORCA"

    ```
    ! SARC-ZORA-TZVPP ZORA DLPNO-CCSD(T) Def2-TZVPP/C TightSCF TightPNO
    %pal nprocs 8
        end
    %Maxcore 15000.00
    %basis
            NewGTO C "ma-ZORA-def2-TZVPP" end
            NewGTO N "ma-ZORA-def2-TZVPP" end
            NewGTO S "ma-ZORA-def2-TZVPP" end
            NewGTO O "ma-ZORA-def2-TZVPP" end
            NewGTO H "ma-ZORA-def2-TZVPP" end
    NewGTO Cl "ma-ZORA-def2-TZVPP" end
    NewGTO Br "ma-ZORA-def2-TZVPP" end
    end
    * xyzfile -1 1 example.xyz
    ```

=== "Gaussian"

    ```
    # wb97xd 6-31++g** Opt=verytight int=ultrafine freq
    ```

Once you have run your analysis, you should have a directory with files named after the related cluster.
For example: [](){ #directory-listing-1 }


<figure markdown="span">
<div style="text-align: left;">
```
2A_2SA_negative.out
2A_1SA_neutral.out
1A_1SA_negative.out
1A_1SA_neutral.out
1A_2SA_negative.out
1SA_negative.out
1A_neutral.out
```
</div>

<figcaption> <em>Figure 1.</em> A directory listing containing only ORCA files</figcaption>
</figure>

It could also be, for example, that you have ORCA `.out` files, Gaussian `.log` files and XYZ `.xyz` like so: [](){ #directory-listing-2 }

<figure markdown="span">
<div style="text-align: left;">
```
2A_2SA_negative.out
2A_2SA_negative.log
2A_2SA_negative.xyz
2A_1SA_neutral.out
2A_1SA_neutral.log
2A_1SA_neutral.xyz
1A_1SA_negative.out
1A_1SA_negative.log
1A_1SA_negative.xyz
1A_1SA_neutral.out
1A_1SA_neutral.log
1A_1SA_neutral.xyz
1A_2SA_negative.out
1A_2SA_negative.log
1A_2SA_negative.xyz
1SA_negative.out
1SA_negative.log
1SA_negative.xyz
1A_neutral.out
1A_neutral.log
1A_neutral.xyz
```
</div>
<figcaption><em>Figure 2.</em> A directory listing containing XYZ, ORCA and Gaussian files</figcaption>
</figure>

You need to create two CSV files: one to contain information about the clusters, and one to contain information about the pathways.
In a moment, we'll talk about how to generate an initial version of these, so read this whole section before getting started.
First however, we'll look at what's in these files.

The clusters CSV relates the cluster name to its associated files.
One option if the files are named like [Figure 2][directory-listing-2] is to have a single column `name` for the cluster name, and another column `prefix` for the prefix of the file name to identify which files relate to which cluster like so:

```
name,prefix
2A_2SA_negative,2A_2SA_negative
2A_1SA_neutral,2A_1SA_neutral

...
```

Another option is to have one column per source file type, and have the file name in each column like so:

```
name,orca,gaussian
2_2SA_negative,gaussian1.out,orca1.log
2A_1SA_neutral,gaussian2.out,orca2.log

...
```

when this is done, you can specify the pathways to consider in the simulation in a pathways CSV file with `parent`, `product1` and `product2` columns like so:

```
parent,product1,product2
2A_2SA_negative,1A_1SA_negative,1A_1SA_neutral
2A_2SA_negative,1A_2SA_negative,1A_neutral
2A_2SA_negative,1SA_negative,2A_1SA_neutral
```

Rather than manually writing these files, you can use the [`apitofsim generate pathways`][apitofsim-generate-pathways] command to generate an initial version of these files based on the files in your directory.
With the `--guess-prefix` option, it will truncate the file extension, assuming your files are laid out as above.
The command needs only the information from an XYZ file, but can attempt to retrieve this from Gaussian or ORCA files using [ASE](https://ase-lib.org/).

So for example, if you have a directory like [Figure 2][directory-listing-2] you could run:

<code>
<a href="../cli/#apitofsim-generate-pathways">apitofsim generate pathways</a> --guess-prefix xyz pathways.csv clusters.csv *.xyz
</code>

And it would generate a `prefix`-based clusters CSV and a pathways CSV with all possible pathways based on the atom counts and charges of the clusters/molecules in the XYZ files.

Alternatively if you had a directory like [Figure 2][directory-listing-1] and wanted a non-prefix based `clusters.csv`, you could run:

<code>
<a href="../cli/#apitofsim-generate-pathways">apitofsim generate pathways</a> orca pathways.csv clusters.csv *.out
</code>

Nested directories are also supported, with the directory name being appended to the name of the cluster in order to prevent ambiguities.
This combines nicely with double star globbing `**` (to recursively traverse directory structures).
In Bash 4+ this can be enabled with `shopt -s globstar` (it is enabled by default in zsh).
For example:

<code>
<a href="../cli/#apitofsim-generate-pathways">apitofsim generate pathways</a> --guess-prefix orca pathways.csv clusters.csv **/*.out
</code>

The next stage is typically to inspect and modify the pathways CSV in case some pathways are not wanted.
This can be done with a text editor or with spreadsheet software such as Excel or LibreOffice Calc.

Once the CSV files are ready, you need to create a TOML configuration.
Assuming you continue working in the same directory, and that you are working with a directory like [Figure 2][directory-listing-2], and that you went with a prefix-based `clusters.csv`, your TOML configuration would start with something like this:

```toml
[[pathways]]
type = "csv"
pathways_path = "pathways.csv"
clusters_path = "clusters.csv"
```

This tells apitofsim you are using CSV ingestion and points to your pathways and clusters CSV files.

The next section tells which sources to use, how to find the filename from the prefix (if needed), and how to combine the data from different sources to get all the needed attributes of the clusters:

```toml
[pathways.clusters]
default_source = "gaussian"
electronic_energy = "orca.final_single_point_energy + gaussian.zero_point_energy"
```

The `default_source` is specified as `"gaussian"`, meaning that unless otherwise specified, the attributes of the clusters will be taken from the Gaussian files.
Each attribute of the following attribute can be specified either as a string specifying a single source to take the attribute from, or as an expression combining attributes from different sources:

 * `rotational_temperatures`
 * `vibrational_temperatures`
 * `electronic_energy`
 * `atomic_mass`
 * `charge`

Finally we specify which source to use as well as the suffix needed to get the full filename from each of the prefixes:

```toml
[pathways.clusters.sources.orca]
append_to_common_prefix = ".out"

[pathways.clusters.sources.gaussian]
append_to_common_prefix = ".log"
```

In case we were not using a prefix-based `clusters.csv`, we would instead specify the full filename in the `clusters.csv` and not need to specify the `append_to_common_prefix` for each source. However the sources still need to be specified like so:

```toml
[pathways.clusters.sources.orca]

[pathways.clusters.sources.gaussian]

```

Next are the parameters for running the simulation, the full description of which is given in [the reference documentation for the configuration file][configuration].
However, the example below should be help with getting started:

```toml
[[configs]]
name = "myconfig"
N_iter = 1_000
M_iter = 1_000
tolerance = 1e-8
resolution = 1_000
N = 1_000
bin_width = "1.0 kelvin"
energy_max = "2.0e5 kelvin"
energy_max_rate = "1.0e5 kelvin"
alpha_factor = "0.25 halfturn"
T = "300.0 kelvin"
dc = "0.0005 meter"
radius_pinhole = "1 mm"
lengths = [ [ 0.001, 0.00244, 0.101, 0.00448, 0.0005 ], "meter" ]
pressures = [ [ 194.0, 3.88 ], "pascal"]
voltages = [ [ -19, -9, -7, -6, 11 ], "volt" ]

[configs.gas]
radius = "1.84e-10 meter"
mass = "4.65e-26 kilogram"
adiabatic_index = 1.4

[configs.quadrupole]
dc_field = "0.0 volt"
ac_field = "200.0 volt"
radiofrequency = "1.3e6 Hz"
r_quadrupole = "6.0e-3 meter"
```

Once these are all prepared you can go ahead and create a database:

<code>
<a href="../cli/#apitofsim-db-prepare">apitofsim db prepare</a> config.toml database.db
</code>

## Running the simulation

Depending on your parameters and the number of pathways, the simulation can take from minutes to hours to run.
The simulation scales well with the number of CPU cores, so you can expect a significant speedup by running on a large node within a computing cluster.
Everything needed is contained in the database, so if you decide to do this, it's simply a matter of installing `apitofsim` on the cluster and then copying over the database file.

Then you are ready to run:

<code>
<a href="../cli/#apitofsim-db-run">apitofsim db run</a> database.db
</code>

There are various options to the [`apitofsim db run`][apitofsim-db-run] command to filter what to run, however the most important option at the moment is `--pathway-at-a-time` which switches between two modes of operation.

 * *The default mode* runs the simulation parent cluster at a time. After each simulated gas collision, the possibility of of fragmenting into any pathway is considered.
 * *Per-pathway mode* runs each pathway separately. After each simulated collision, only one pathway is considered.

## Analysing the results

All results are stored in the database.
You can generate CSV reports from the database file, e.g. for usage in Excel or LibreOffice Calc, with the `apitofsim db report` command, for example:

<code>
<a href="../cli/#apitofsim-db-report">apitofsim db report</a pathway-report database.db pathway_report.csv
<a href="../cli/#apitofsim-db-report">apitofsim db report</a experiment-report database.db experiment_report.csv
<a href="../cli/#apitofsim-db-report">apitofsim db report</a> experiment-summary database.db experiment_summary.csv
</code>

You can also generate plots directly from the database file.

As a summary, you can plot a bar char of the survival probability of each cluster:

<code>
<a href="../cli/#apitofsim-db-plot-survival">apitofsim db plot survival</a> database.db survival.png
</code>

You can also generate spectrograms.
For example:

<code>
<a href="../cli/#apitofsim-db-plot-spectrogram-many">apitofsim db spectrogram-many</a> database.db spectrograms
</code>

This will create a directory called `spectrograms` containing a spectrogram for each cluster.

Note that the fragmentation simulation does not take into account transmission efficiency.
There are some rudimentary models available which can be applied in order to attempt to model transmission based on mass.
These are based simple curve fitting based on experimental data.
They correspond to particular instrument settings, and are expected to be even less accurate when used for other settings.

| Name          | Charge   | Voltages (V) |
| ------------- | ---------|--------------|
| `old`         | negative | $[-19, -9, -7, -6, 11]$ |
| `new_neg`     | negative | $[-19, -9, -7, -6, 11]$ |
| `new_pos`     | positive | $[25, 12, -4, -8, -17]$ |

You can use these models for plotting spectrograms like so:

<code>
<a href="../cli/#apitofsim-db-plot-spectrogram-many">apitofsim db spectrogram-many</a> --model-transmission=new_neg database.db spectrograms
</code>
