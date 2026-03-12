from typing import Any, NoReturn


class CombineError(Exception):
    def __init__(self, info):
        super().__init__()
        self.info = info


def import_source(
    source,
    method,
    particle_name,
    source_name,
    cluster_info,
    path_base,
    *,
    ingest_ase=False,
):
    from ase.io import read as ase_read

    ignore_unicode_errors = source.get("ignore_unicode_errors", False)
    if ignore_unicode_errors:
        error_handler = "ignore"
    else:
        error_handler = "strict"

    if method in ("orca", "gaussian", "xyz"):
        if hasattr(cluster_info, source_name):
            path = getattr(cluster_info, source_name)
        elif hasattr(cluster_info, "prefix"):
            extension = source["append_to_common_prefix"]
            path = path_base / (cluster_info.prefix + extension)
        else:
            raise ValueError(
                f"Source {source_name} for particle {particle_name} does not have a path column, "
                "and the cluster CSV does not have a prefix to construct one from"
            )
        if method == "orca":
            from apitofsim.ingest.orca import parse_orca

            with open(path, errors=error_handler) as f:
                orca_result = parse_orca(f)
                if len(orca_result) != 1:
                    raise ValueError(
                        f"Expected one structure in ORCA output {path}, got {len(orca_result)}"
                    )
                result = orca_result[0]
                if ingest_ase:
                    f.seek(0)
                    try:
                        result["ase"] = ase_read(f, format="orca-output")
                    except Exception as e:
                        result["ase"] = e
        elif method == "gaussian":
            from apitofsim.ingest.gaussian import parse_gaussian

            with open(path, errors=error_handler) as f:
                result = parse_gaussian(f)
                if ingest_ase:
                    f.seek(0)
                    try:
                        result["ase"] = ase_read(f, format="gaussian-out")
                    except Exception as e:
                        result["ase"] = e
        else:
            assert method == "xyz"
            result = {}
            if ingest_ase:
                try:
                    result["ase"] = ase_read(path, format="xyz")
                except Exception as e:
                    result["ase"] = e
        return result
    elif method == "map":
        return source.get(particle_name, {})
    else:
        raise ValueError(f"Unknown method: {method}")


def import_sources(
    clusters,
    particle_name,
    cluster_info,
    path_base,
    *,
    ingest_ase=False,
    ignore_unicode_errors=False,
):
    sources = {}
    for source_name, source in clusters["sources"].items():
        method = source.get("type", source_name)
        sources[source_name] = import_source(
            source,
            method,
            particle_name,
            source_name,
            cluster_info,
            path_base,
            ingest_ase=ingest_ase,
        )
    return sources


class DotAccessDict(dict[str, Any]):
    __getattr__ = dict.get


def combine_sources(sources, clusters, *, ingest_ase=False):
    provenance = {}
    combined = {}
    for quantity in [
        "number_of_atoms",
        "vibrational_temperatures",
        "rotational_temperatures",
        "electronic_energy",
        "atomic_mass",
        "charge",
        *(("ase",) if ingest_ase else ()),
    ]:
        use_eval = False
        if quantity in clusters:
            source_specifier = quantity
        else:
            source_specifier = "default_source"
        source_name = clusters[source_specifier]
        if "." in source_name:
            if quantity == "ase":
                raise ValueError("Cannot use programmatic source for ASE information")
            use_eval = True

        def raise_combine_error(e) -> NoReturn:
            available_source_quantities = [
                f"{source_name}.{quantity}"
                for source_name, source in sources.items()
                for quantity in source.keys()
            ]
            raise CombineError(
                {
                    "source_name": source_name,
                    "source_specifier": source_specifier,
                    "quantity": quantity,
                    "exception": e,
                    "available_source_quantities": available_source_quantities,
                }
            )

        if use_eval:
            eval_ctx = {
                source_name: DotAccessDict(sources[source_name])
                for source_name in sources
            }
            try:
                result = eval(source_name, eval_ctx)
            except Exception as e:
                raise_combine_error(e)
        else:
            try:
                source_results = sources[source_name]
                if (
                    quantity == "vibrational_temperatures"
                    and quantity not in source_results
                    and combined["number_of_atoms"] == 1
                ):
                    # Probably an atomic-like product
                    result = None
                else:
                    result = source_results[quantity]
            except Exception as e:
                raise_combine_error(e)
            if quantity == "ase" and isinstance(result, Exception):
                raise result
        combined[quantity] = result
        provenance[quantity] = source_name
    return combined, provenance
