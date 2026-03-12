from ase.io.orca import get_chunks as read_orca_chunks
from numpy import array
from pint import get_application_registry

ureg = get_application_registry()


def parse_orca(fd):
    chunks = list(read_orca_chunks(fd))
    out_chunks = []
    for chunk in chunks:
        out_chunk = {}
        lines_iter = iter(chunk)
        while 1:
            try:
                line = next(lines_iter)
            except StopIteration:
                break
            version_marker = "Program Version"
            input_file_marker = "INPUT FILE"
            fspe_marker = "FINAL SINGLE POINT ENERGY"
            rot_marker = "Rotational constants in cm-1:"
            charge_marker = " Total Charge           Charge          ...."
            temperature_marker = "Temperature         ..."
            pressure_marker = "Pressure            ..."
            total_mass_marker = "Total Mass          ..."
            number_of_atoms_marker = "Number of atoms                             ..."
            line_bare = line.strip()
            if line_bare.startswith(version_marker):
                out_chunk["software_version"] = line_bare
            elif line_bare.startswith(input_file_marker):
                next(lines_iter)  # Skip ===
                while 1:
                    line = next(lines_iter).rstrip()
                    if line.startswith("==="):
                        break
                    out_chunk.setdefault("input", []).append(line)
            elif line.startswith(fspe_marker):
                out_chunk["final_single_point_energy"] = ureg.Quantity(
                    float(line[len(fspe_marker) :].strip()), "hartree"
                )
            elif "Zero point energy" in line:
                energy = float(line.split("...")[-1].strip().split()[0])
                out_chunk["zero_point_energy"] = ureg.Quantity(energy, "hartree")
            elif line.startswith(rot_marker):
                out_chunk["rotational_temperatures"] = ureg.Quantity(
                    array([float(n) for n in line[len(rot_marker) :].strip().split()]),
                    "reciprocal_centimeter",
                )
            elif "E(vib)   ..." in line:
                out_chunk.setdefault("vibrational_temperatures", []).append(
                    float(line.split("...")[0].strip().split()[-2].strip())
                )
            elif line.startswith(charge_marker):
                out_chunk["charge"] = int(line.split("....")[-1].strip())
            elif line.startswith(temperature_marker):
                out_chunk["temperature"] = ureg.Quantity(
                    float(line.split("...")[-1].strip().split()[0]), "kelvin"
                )
            elif line.startswith(pressure_marker):
                out_chunk["pressure"] = (
                    float(line.split("...")[-1].strip().split()[0]) * ureg.atmosphere
                )
            elif line.startswith(total_mass_marker):
                out_chunk["atomic_mass"] = ureg.Quantity(
                    float(line.split("...")[-1].strip().split()[0]), "amu"
                )
            elif line.startswith(number_of_atoms_marker):
                out_chunk["number_of_atoms"] = int(line.split("...")[-1].strip())
        if "vibrational_temperatures" in out_chunk:
            out_chunk["vibrational_temperatures"] = ureg.Quantity(
                array(out_chunk["vibrational_temperatures"]), "reciprocal_centimeter"
            )
        if input in out_chunk:
            out_chunk["input"] = "\n".join(out_chunk["input"])
        out_chunks.append(out_chunk)
    return out_chunks
