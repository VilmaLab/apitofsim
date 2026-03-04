from pint import get_application_registry

from .db import ClusterDatabase

ureg = get_application_registry()


def insert_parsed_pathway(db, pathway, *, prefix=None):
    from apitofsim.config import dump_to_raw

    ids = []
    for particle_info in pathway:
        name = particle_info["name"]
        if prefix is not None:
            name = prefix + name
        combined = particle_info["particle"]
        with ureg.context("boltzmann", "spectroscopy"):
            inserted, id = db.insert_cluster(
                name,
                combined["atomic_mass"].to("amu").magnitude,
                combined["charge"],
                combined["electronic_energy"].to("hartree").magnitude,
                combined["rotational_temperatures"].to("K").magnitude
                if combined["rotational_temperatures"] is not None
                else None,
                combined["vibrational_temperatures"].to("K").magnitude
                if combined["vibrational_temperatures"] is not None
                else None,
                dump_to_raw(particle_info).decode("utf-8"),
                allow_duplicates=True,
            )
            ids.append(id)
    db.insert_pathway(*ids)


def ingest_legacy_one(
    db: ClusterDatabase, filename, clusters, prefix=None, path_base=None
):
    from apitofsim.ingest.legacy import parse_legacy_one

    pathway = parse_legacy_one(filename, clusters, path_base=path_base)
    insert_parsed_pathway(db, pathway, prefix=prefix)


def ingest_tree(db: ClusterDatabase, pathways, path_base, descriptor=None):
    if isinstance(pathways, list):
        for idx, pathways_segment in enumerate(pathways):
            ingest_tree(db, pathways_segment, path_base, (descriptor, idx))
        return
    if pathways["type"] == "legacy_glob":
        from apitofsim.ingest.legacy import parse_legacy_tree

        for pathway in parse_legacy_tree(
            pathways["path"], pathways["clusters"], path_base
        ):
            insert_parsed_pathway(db, pathway, prefix=pathways.get("prefix"))
    elif pathways["type"] == "csv":
        from apitofsim.ingest.csv import parse_csv_tree

        for pathway in parse_csv_tree(
            path_base / pathways["pathways_path"],
            path_base / pathways["clusters_path"],
            pathways["clusters"],
            path_base,
            descriptor,
        ):
            insert_parsed_pathway(db, pathway, prefix=pathways.get("prefix"))
