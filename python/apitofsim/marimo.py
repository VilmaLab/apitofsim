import marimo as mo


def get_histogram_ids(db):
    return db.db.execute(
        """
        with dos_histogram_ids as (
            select histogram_params_id, count(*) as dos_count from cluster_dos group by histogram_params_id
            union
            select histogram_params_id, count(*) as dos_count from products_dos group by histogram_params_id
        )
        select histogram_params_id, bin_width, max, dos_histogram_ids.dos_count
        from histogram_params
        join dos_histogram_ids
        on dos_histogram_ids.histogram_params_id = histogram_params.id
        order by bin_width
        """
    ).fetchall()


def dos_histogram_dropdown(db):
    histogram_ids = get_histogram_ids(db)
    return mo.ui.dropdown(
        options={
            f"Histogram for DOS with bin width: {bin_width}, max: {max} and {dos_count} entries)": (
                histogram_params_id,
                bin_width,
                max,
                dos_count,
            )
            for (histogram_params_id, bin_width, max, dos_count) in histogram_ids
        },
        label="Pick a DOS histogram",
    )


def cluster_dropdown(name_lookup):
    return mo.ui.dropdown(
        options={v: k for k, v in name_lookup.items()},
        label="Pick a cluster",
        searchable=True,
    )
