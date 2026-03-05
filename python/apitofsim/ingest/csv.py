from os.path import expanduser
from typing import Any

import pandas

from .common import CombineError, combine_sources, import_sources


def parse_csv_tree(
    pathways_path, clusters_path, clusters_config, path_base, descriptor=None
):
    clusters_df = pandas.read_csv(expanduser(clusters_path))
    # TODO: Validate columns of clusters_df
    cluster_dict: dict[str, Any] = {}
    for cluster_info in clusters_df.itertuples():
        if not hasattr(cluster_info, "name"):
            raise ValueError(
                "Expected column 'name' in clusters CSV, but it was not found"
            )
        particle_name = cluster_info.name  # pyright: ignore[reportAttributeAccessIssue]
        sources = import_sources(
            clusters_config,
            particle_name,
            cluster_info,  # pyright: ignore[reportAttributeAccessIssue]
            path_base,
        )
        try:
            combined, provenance = combine_sources(sources, clusters_config)
        except CombineError as e:
            e.info["particle_name"] = particle_name
            source_specifier = e.info["source_specifier"]
            path = f"pathways.clusters.{source_specifier}"
            e.info["path"] = path
            if descriptor is not None:
                if isinstance(descriptor, tuple):
                    descriptor, idx = descriptor
                    field_descriptor = descriptor.get_field_from_aot(path)[idx]
                else:
                    field_descriptor = descriptor.get_field(path)
                e.info["line_no"] = field_descriptor.line_no
            raise e
        cluster_dict[particle_name] = {
            "name": particle_name,
            "sources": sources,
            "particle": combined,
            "provenance": provenance,
        }

    pathways_df = pandas.read_csv(expanduser(pathways_path))
    # TODO: Validate columns of pathways_df
    for tpl in pathways_df.itertuples():
        for attr in ["parent", "product1", "product2"]:
            if not hasattr(tpl, attr):
                raise ValueError(
                    "Expected column '{attr}' in pathways CSV, but it was not found"
                )
        pathway = []
        for particle_name in [tpl.parent, tpl.product1, tpl.product2]:  # pyright: ignore[reportAttributeAccessIssue]
            pathway.append(cluster_dict[particle_name])
        yield pathway
