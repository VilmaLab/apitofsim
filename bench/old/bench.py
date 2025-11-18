import resource
from os import makedirs, environ, listdir
from os.path import join as pjoin
from subprocess import check_output, STDOUT, CalledProcessError
from shutil import copyfile
import pickle
import time
from re import search

def mkp(path):
    makedirs(path, exist_ok=True)

VMS_BIN = environ["VMS_BIN"]
BASE_INPUTS = environ["BASE_INPUTS"]
BENCHWORK = environ.get("BENCHWORK", "benchwork")
COMPILERS = environ.get("COMPILERS", "").split(",")
MAX_NUM_THREADS = int(environ.get("MAX_NUM_THREADS", 8))
NUM_REALIZATIONS = 100

def sayrun(prog, inp, **kwargs):
    try:
        print("Running:", prog)
        return check_output([prog], input=inp, text=True, stderr=STDOUT, **kwargs)
    except CalledProcessError as e:
        print()
        print(" *** ")
        print(f"Error running command: {e.cmd}")
        print(f"Output: {e.output}")
        print()
        raise


def run_precompiler(compiler):
    pre_compiler = environ.get(f"PRE_COMPILER_{compiler.upper()}", "")
    if pre_compiler:
        print(f"Running pre-compiler command for {compiler}")
        sayrun(pre_compiler, "", shell=True)
    else:
        print(f"No pre-compiler command set for {compiler}, skipping.")


def make_config(outdir, realizations):
    return f"""
    GENERAL_INPUT_FOR_APITOF_CODE
    -1                      Cluster_charge_sign    
    216                 Atomic_mass_cluster
    136           Atomic_mass_first_product
    80          Atomic_mass_second_product
    300.0                         Temperature_(K)
    182.0                         Pressure_first_chamber(Pa)
    3.53                          Pressure_second_chamber(Pa)
    1.0e-3                        Length_of_1st_chamber_(meters)
    5.0e-4                        Length_of_skimmer_(meters)
    2.44e-3                       Length_between_skimmer_and_front_quadrupole_(meters)
    0.101                         Length_between_front_quadrupole_and_back_quadrupole_(meters)
    4.48e-3                       Length_between_back_quadrupole_and_2nd_skimmer_(meters)
    -19                       Voltage0_(Volt)
    -9                       Voltage1_(Volt)
    -7                       Voltage2_(Volt)
    -6                       Voltage3_(Volt)
    11                       Voltage4_(Volt)
    {realizations}                Number_of_realizations
    5.0e-4                        Radius_at_smallest_cross_section_skimmer_(m)
    0.25                          Angle_of_skimmer_(multiple_of_PI)
    0                             Fragmentation_energy_(Kelvin)
    2.0e5                         Energy_max_for_density_of_states_(Kelvin)
    3.0e4                         Energy_max_for_rate_constant_(Kelvin)
    1.0                           Energy_resolution_(Kelvin)
    2.46e-10                      Gas_molecule_radius_(meters)
    4.8506e-26                    Gas_molecule_mass_(kg)
    1.4                           Adiabatic_index
    0.0                           DC_quadrupole
    200.0                         AC_quadrupole
    1.3e6                         Radiofrequency_quadrupole
    6.0e-3                        Half-distance_between_quadrupole_rods
    {outdir}/skimmer.dat    Output_file_skimmer
    {BASE_INPUTS}/1ABisopooh1brd1w-1100001000_1_129_vib.dat                 file_vibrational_temperatures_cluster
    {BASE_INPUTS}/1ABisopooh1w-1010000_7_18-str7-str7_vib.dat                 file_vibrational_temperatures_first_product
    {BASE_INPUTS}/1brd-1000_1_0_vib.dat                 file_vibrational_temperatures_second_product
    {BASE_INPUTS}/1ABisopooh1brd1w-1100001000_1_129_rot.dat                 file_rotational_temperatures_cluster
    {BASE_INPUTS}/1ABisopooh1w-1010000_7_18-str7-str7_rot.dat                 file_rotational_temperatures_first_product
    {BASE_INPUTS}/1brd-1000_1_0_rot.dat                 file_rotational_temperatures_second_product
    {BASE_INPUTS}/1ABisopooh1brd1w-1100001000_1_129_en.dat                 file_electronic_energy_cluster
    {BASE_INPUTS}/1ABisopooh1w-1010000_7_18-str7-str7_en.dat                 file_electronic_energy_first_product
    {BASE_INPUTS}/1brd-1000_1_0_en.dat                 file_electronic_energy_second_product
    {outdir}/density_cluster.out        output_file_density_cluster
    {outdir}/density_first_product.out  output_file_density_first_product
    {outdir}/density_second_product.out output_file_density_second_product
    {outdir}/density_combined_product.out output_file_density_combined_products
    {outdir}/rate_constant.out          output_file_rate_constant
    {outdir}/survivals.dat              output_file_probabilities
    1000                          Number_of_iterations_in_solving_equation
    1000                          Number_of_iterations_in_solving_equation2
    1000                          Number_of_solved_points
    1.0e-8                        Tolerance_in_solving_equation
    """.strip()

common_out = pjoin(BENCHWORK, "commonout")
mkp(common_out)

common_config = make_config(common_out, NUM_REALIZATIONS)
print("** Begin common config **")
print(common_config)
print("** End common config **")

run_precompiler("gcc")
sayrun(pjoin(VMS_BIN, "skimmer_win"), common_config)
sayrun(pjoin(VMS_BIN, "densityandrate_win"), common_config)

times = {}
for compiler in COMPILERS:
    if compiler:
        run_precompiler(compiler)
    for cores in range(1, MAX_NUM_THREADS + 1):
        print(f"Running compiler {compiler} and cores {cores}")
        out = pjoin(BENCHWORK, f"output_{cores}" + (f"_{compiler}" if compiler else ""))
        mkp(out)
        for fn in listdir(common_out):
            copyfile(pjoin(common_out, fn), pjoin(out, fn))
        config = make_config(out, NUM_REALIZATIONS)
        print(f"** Begin config for {compiler} with {cores} cores **")
        print(config)
        print(f"** End config for {compiler} with {cores} cores **")
        bin_path = pjoin(VMS_BIN, "apitof_pinhole")
        if compiler:
            bin_path += "." + compiler
        # Can't run this in HPC. TODO: Bug them about it.
        #sayrun("sync; echo 3 | sudo tee /proc/sys/vm/drop_caches", shell=True)
        time_start = time.time()
        usage_start = resource.getrusage(resource.RUSAGE_CHILDREN)
        output = sayrun(bin_path, config, env={"OMP_NUM_THREADS": str(cores)})
        match = search(r"<loop_time>([^<]+)</loop_time>", output)
        usage_end = resource.getrusage(resource.RUSAGE_CHILDREN)
        time_end = time.time()
        cpu_time = usage_end.ru_utime - usage_start.ru_utime
        sys_time = usage_end.ru_stime - usage_start.ru_stime
        wall_time = time_end - time_start
        key = (
            ("cores", cores),
        )
        if compiler:
            key += (("compiler", compiler))
        times[key] = {
            "cpu_time": cpu_time,
            "sys_time": sys_time,
            "wall_time": wall_time,
            "loop_time": float(match.group(1)) if match else None,
            "iters_per_second": NUM_REALIZATIONS * 1_000_000 / float(match.group(1)) if match else None,
        }

with open("times.pkl", "wb") as outf:
    pickle.dump(times, outf)