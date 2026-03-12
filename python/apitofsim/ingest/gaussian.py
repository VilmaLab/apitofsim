from numpy import array
from pint import get_application_registry

ureg = get_application_registry()


def parse_gaussian(fd):
    out = {}
    lines = iter(fd)
    while 1:
        try:
            line = next(lines)
        except StopIteration:
            break
        charge_marker = " Charge ="
        frequencies_marker = " Frequencies -- "
        zpc_marker = " Zero-point correction="
        cite_marker = " Cite this work as:"
        rotational_temperatures_marker = " Rotational temperatures (Kelvin) "
        mass_marker = " Molecular mass: "
        if line.startswith(charge_marker):
            out["charge"] = int(line[len(charge_marker) :].strip().split()[0])
        elif line.startswith(frequencies_marker):
            new_freqs = (
                float(x) for x in line[len(frequencies_marker) :].strip().split()
            )
            out.setdefault("vibrational_temperatures", []).extend(new_freqs)
        elif line.startswith(rotational_temperatures_marker):
            lst = [
                float(x)
                for x in line[len(rotational_temperatures_marker) :].strip().split()
            ]
            lst.reverse()
            out["rotational_temperatures"] = ureg.Quantity(array(lst), "kelvin")
        elif line.startswith(" Temperature "):
            bits = line.strip().split()
            out["temperature"] = ureg.Quantity(float(bits[1]), "kelvin")
            out["pressure"] = float(bits[4]) * ureg.atmosphere
        elif line.startswith(zpc_marker):
            out["zero_point_energy"] = ureg.Quantity(
                float(line[len(zpc_marker) :].strip().split()[0]), "hartree"
            )
        elif line.startswith(mass_marker):
            mass = float(line[len(mass_marker) :].strip().split()[0])
            out["atomic_mass"] = ureg.Quantity(mass, "amu")
        elif line.startswith(" NAtoms="):
            out["number_of_atoms"] = int(line.strip().split()[1])
        elif line.startswith(cite_marker):
            cite = []
            while 1:
                # Grab version info
                line = next(lines).strip()
                if line == "":
                    break
                cite.append(line)
            out["citation"] = "\n".join(cite)
            next(lines)  # Skip ***
            version_and_date = next(lines).strip() + "\n" + next(lines).strip()
            out["version_and_date"] = version_and_date
            next(lines)  # ***
            while 1:
                line = next(lines).strip()
                if line.startswith("#"):
                    out["input"] = line
                    break
    if "vibrational_temperatures" in out:
        out["vibrational_temperatures"] = ureg.Quantity(
            array(out["vibrational_temperatures"]), "reciprocal_centimeter"
        )
    return out
