from typing import Any, NoReturn


class CombineError(Exception):
    def __init__(self, info):
        super().__init__()
        self.info = info


def import_source(source, method, particle_name, prefix, path_base):
    if method == "orca":
        from apitofsim.ingest.orca import parse_orca

        extension = source["append_to_common_prefix"]
        path = path_base / (prefix + extension)
        with open(path) as f:
            orca_result = parse_orca(f)
            if len(orca_result) != 1:
                raise ValueError(
                    f"Expected one structure in ORCA output {path}, got {len(orca_result)}"
                )
            return orca_result[0]
    elif method == "gaussian":
        from apitofsim.ingest.gaussian import parse_gaussian

        extension = source["append_to_common_prefix"]
        path = path_base / (prefix + extension)
        with open(path) as f:
            return parse_gaussian(f)
    elif method == "map":
        return source.get(particle_name, {})
    else:
        raise ValueError(f"Unknown method: {method}")


def import_sources(clusters, particle_name, prefix, path_base):
    sources = {}
    for source_name, source in clusters["sources"].items():
        method = source.get("type", source_name)
        sources[source_name] = import_source(
            source, method, particle_name, prefix, path_base
        )
    return sources


class DotAccessDict(dict[str, Any]):
    __getattr__ = dict.get


def combine_sources(sources, clusters):
    provenance = {}
    combined = {}
    for quantity in [
        "vibrational_temperatures",
        "rotational_temperatures",
        "electronic_energy",
        "atomic_mass",
        "charge",
    ]:
        use_eval = False
        if quantity in clusters:
            source_specifier = quantity
        else:
            source_specifier = "default_source"
        source_name = clusters[source_specifier]
        if "." in source_name:
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
                result = sources[source_name][quantity]
            except Exception as e:
                raise_combine_error(e)
        combined[quantity] = result
        provenance[quantity] = source_name
    return combined, provenance
