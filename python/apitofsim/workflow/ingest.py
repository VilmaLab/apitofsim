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


def ingest_legacy_one(db: ClusterDatabase, filename, clusters, prefix=None):
    from apitofsim.ingest.legacy import parse_legacy_one

    pathway = parse_legacy_one(filename, clusters)
    insert_parsed_pathway(db, pathway, prefix=None)


def ingest_tree(db: ClusterDatabase, pathways):
    if isinstance(pathways, list):
        for pathways_segment in pathways:
            ingest_tree(db, pathways_segment)
        return
    if pathways["type"] == "legacy_glob":
        from apitofsim.ingest.legacy import parse_legacy_tree

        for pathway in parse_legacy_tree(pathways["path"], pathways["clusters"]):
            insert_parsed_pathway(db, pathway, prefix=pathways.get("prefix"))
