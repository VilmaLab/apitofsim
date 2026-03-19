# Background on the APi-TOF simulation

This page gives some background information on how the package works and gives references to relevant publications.

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
This publication describes the simulation of the skimmer and quadrupole.
The code used in this publication is an earlier version of the code in this repository.

Later, Zanca, T. (2025)[^4] added support for atom-like products.

The current version of the code supports sampling schemes other than the originally described histogram-based technique. The rejection sampling method is described [in one of the notebooks included in the source code repository](notebooks.md).

Two approaches to considering multiple fragmentation pathways, e.g. in order to obtain the intensities of a spectrogram, have been added.
In the first and default, the simulation considers multiple pathways at once.
In particular, after a collision with a gas particle, survival probability distributions are considered for all pathways, and the first to occur after sampling determines the fragmentation pathways along which the fragmentation occurs in the case that another a simulated gas collision does not happen first.
In the second, corresponding to earlier versions of the code, the simulation considers each pathway separately and uses assumptions of independence to obtain the overall survival probability as the product of the survival probabilities of each pathway.
Beyond this, per-pathway fragmentation probabilities can be found using independence and assuming fragmentation happens in an order proportional to some measure of overall pathway fragmentation as outlined by Zanca, T. (2025)[^5].
This second approach is available through the `--pathway-at-a-time` option.

The software has some additional features not yet described in publication or via a notebook. They are:

* Support for either negatively or positively charged clusters.
* Pinhole rejection: Extra code to compensate for reduced gas collision frequency near to the pinhole.
* Modelling of the skimmer using a quasi-one-dimensional isentropic nozzle flow calculation.

[^1]:
    Zapadinsky, E., Passananti, M., Myllys, N., Kurtén, T., & Vehkamäki, H. (2019).
    Modeling on Fragmentation of Clusters inside a Mass Spectrometer.
    *The Journal of Physical Chemistry. A*, 123, 611 - 624.
    [[web]](https://pubs.acs.org/doi/10.1021/acs.jpca.8b10744) [[pdf]](https://pubs.acs.org/doi/pdf/10.1021/acs.jpca.8b10744?ref=article_openPDF) [[doi]](https://doi.org/10.1021/acs.jpca.8b10744)
[^2]:
    Zapadinsky, E., Passananti, M., Myllys, N., Kurtén, T., & Vehkamäki, H. (2019).
    Supporting Information to "Modelling on Fragmentation of Clusters Inside a Mass Spectrometer"
    [[pdf]](https://pubs.acs.org/doi/suppl/10.1021/acs.jpca.8b10744/suppl_file/jp8b10744_si_001.pdf)
[^3]:
    Zanca, T., Kubečka, J., Zapadinsky, E., Passananti, M., Kurtén, T., & Vehkamäki, H. (2020).
    Highly oxygenated organic molecule cluster decomposition in atmospheric pressure interface time-of-flight mass spectrometers.
    *Atmospheric Measurement Techniques*, 13, 3581-3593.
    [[web]](https://amt.copernicus.org/articles/13/3581/2020/) [[pdf]](https://amt.copernicus.org/articles/13/3581/2020/amt-13-3581-2020.pdf) [[doi]](https://doi.org/10.5194/amt-13-3581-2020)
[^4]:
    Zanca, T. (2025).
    Fragmentation rate constant for an atom-like product.
    *Note published online.*
    [[web]](atom-like_product.pdf)
[^5]:
    Zanca, T. (2025).
    Single and multi-pathway fragmentations.
    *Note published online.*
    [[web]](single-and-multi-pathway-fragmentations.pdf)

## Publications using the simulation

These publication make use (previous versions of) this simulation:

* Alfaouri et al. (2022)[^6] used an older version of the code, without support for the quadrupole, pinhole rejection or multiple fragmentation pathways.
  In this publication, the overall rejection rate is obtained as the product of the rejection rate of individual pathways.
  The [workflow example](python.md) builds on this study.

[^6]:
    Alfaouri, D., Passananti, M., Zanca, T., Ahonen, L., Kangasluoma, J., Kubečka, J., Myllys, N., and Vehkamäki, H. (2022).
    A study on the fragmentation of sulfuric acid and dimethylamine clusters inside an atmospheric pressure interface time-of-flight mass spectrometer.
    *Atmospheric Measurement Techniques*, 15, 11–19.
    [[web]](https://amt.copernicus.org/articles/15/11/2022/) [[pdf]](https://amt.copernicus.org/articles/15/11/2022/amt-15-11-2022.pdf) [[doi]](https://doi.org/10.5194/amt-15-11-2022)
