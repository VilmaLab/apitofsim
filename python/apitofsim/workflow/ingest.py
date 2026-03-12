from pint import get_application_registry

from .db import ClusterDatabase

ureg = get_application_registry()


def insert_parsed_pathway(db, pathway, *, prefix=None):
    from apitofsim.config import dump_to_raw

    ids = []
    for particle_info in pathway:
        from pprint import pprint

        pprint(particle_info)
        name = particle_info["name"]
        if prefix is not None:
            name = prefix + name
        combined = particle_info["particle"]
        with ureg.context("boltzmann", "spectroscopy"):
            inserted, cluster_id = db.insert_cluster(
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
            if inserted and "ase" in combined:
                db.insert_ase(cluster_id, combined["ase"])
            ids.append(cluster_id)
    db.insert_pathway(*ids)


def ingest_legacy_one(
    db: ClusterDatabase, filename, clusters, prefix=None, path_base=None
):
    from apitofsim.ingest.legacy import parse_legacy_one

    pathway = parse_legacy_one(filename, clusters, path_base=path_base)
    insert_parsed_pathway(db, pathway, prefix=prefix)


def ingest_tree(
    db: ClusterDatabase, pathways, path_base, descriptor=None, ingest_ase=None
):
    """
    Ingest a tree of fragmentation pathways into the database.

     * `pathways` can be a list or single element, and matches the configuration

     * `path_base` is used to resolve relative paths in the config

     * `descriptor` is used in case the config is parsed from TOML,
       to provide line numbers for errors,
       and so not typically used outside apitofsim itself.

     * `ingest_ase` is a boolean that controls whether to ingest ASE information.
       By default, it will be be true iff you pass a `ClusterDatabase`.
    """
    if ingest_ase is None:
        ingest_ase = db.ase_db is not None

    if isinstance(pathways, list):
        for idx, pathways_segment in enumerate(pathways):
            ingest_tree(
                db,
                pathways_segment,
                path_base,
                (descriptor, idx),
                ingest_ase=ingest_ase,
            )
        return
    if pathways["type"] == "legacy_glob":
        if ingest_ase:
            raise ValueError(
                "Legacy glob pathway ingestion does not support ASE ingestion (ingest_ase=True)"
            )
        from apitofsim.ingest.legacy import parse_legacy_tree

        for pathway in parse_legacy_tree(
            pathways["path"], pathways["clusters"], path_base
        ):
            insert_parsed_pathway(db, pathway, prefix=pathways.get("prefix"))
    elif pathways["type"] == "csv":
        from apitofsim.ingest.csv import parse_csv_tree

        clusters_path = path_base / pathways["clusters_path"]
        current_path_base = clusters_path.parent

        for pathway in parse_csv_tree(
            path_base / pathways["pathways_path"],
            clusters_path,
            pathways["clusters"],
            current_path_base,
            descriptor=descriptor,
            ingest_ase=ingest_ase,
        ):
            insert_parsed_pathway(db, pathway, prefix=pathways.get("prefix"))
