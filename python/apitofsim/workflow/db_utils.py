import duckdb


def duckdb_connect_roview_cow(filename, *, config=None, fallback="copy"):
    import os
    from os.path import join as pjoin
    from os.path import split as psplit
    from os.path import splitext
    from shutil import copy
    from uuid import uuid4

    import duckdb

    if config is None:
        config = {}
    path, base = psplit(filename)
    base, ext = splitext(base)
    rnd = uuid4().hex
    dest = pjoin(path, f".{base}.rosnap.{rnd}.{ext}")
    exc = None
    cleanup = None
    try:
        from reflink import (  # pyright: ignore[reportMissingImports]
            ReflinkImpossibleError,
            reflink,
        )

        try:
            reflink(filename, dest)
        except (NotImplementedError, ReflinkImpossibleError) as e:
            exc = e
        else:
            cleanup = dest
    except ModuleNotFoundError as e:
        exc = e
    if exc is not None:
        if fallback == "copy":
            copy(filename, dest)
            cleanup = dest
        elif fallback == "connect":
            dest = filename
        elif fallback == "error":
            raise exc
        else:
            raise ValueError(f"Invalid fallback option: {fallback}")
    db = duckdb.connect(dest, read_only=True, config=config)
    cleanup_fn = None
    if cleanup is not None:

        def _cleanup_fn():
            os.remove(cleanup)

        cleanup_fn = _cleanup_fn
    return db, cleanup_fn


def get_or_insert(db, tbl, **vals):
    query = db.table(tbl)
    for col_name, col_value in vals.items():
        query = query.filter(
            duckdb.ColumnExpression(col_name) == duckdb.ConstantExpression(col_value)
        )
    result = list(query.fetchall())
    if len(result) > 1:
        raise ValueError("Multiple rows found")
    if len(result) == 1:
        return result[0][0]
    expression = (
        "insert into "
        + tbl
        + " ("
        + ",".join(vals.keys())
        + ") "
        + "values ("
        + ",".join("?" for _ in vals)
        + ") returning id"
    )
    id = db.execute(expression, tuple(vals.values())).fetchone()
    return id[0]


def get_through_join_else(conn, rel, proj_col, result_dict, **match_cols):
    import pyarrow as pa

    iterable_match_keys = []
    scalar_match_keys = []
    for match_key, match_col in match_cols.items():
        if hasattr(match_col, "__iter__"):
            iterable_match_keys.append(match_key)
        else:
            scalar_match_keys.append(match_key)

    iterable_match_vals = [match_cols[k] for k in iterable_match_keys]
    wanted_tbl = pa.table(iterable_match_vals, names=iterable_match_keys)
    condition = None
    for k in iterable_match_keys:
        new = duckdb.ColumnExpression(f"wanted.{k}") == duckdb.ColumnExpression(
            f"rel.{k}"
        )
        if condition is None:
            condition = new
        else:
            condition = condition & new
    for k in scalar_match_keys:
        new = duckdb.ColumnExpression(f"rel.{k}") == duckdb.ConstantExpression(
            match_cols[k]
        )
        if condition is None:
            condition = new
        else:
            condition = condition & new
    data = (
        rel.set_alias("rel")
        .join(
            conn.from_arrow(wanted_tbl).set_alias("wanted"),
            condition=condition,
            how="right",
        )
        .select(proj_col)
        .to_arrow_table()
    )
    try:
        data = data.column(proj_col).chunk(0)
    except IndexError:
        return
    for match_row, value in zip(zip(*iterable_match_vals), data):
        value = value.values
        if value is not None:
            # Hit: update result_dict
            data = value.to_numpy(zero_copy_only=True)
            if len(match_row) == 1:
                match_row = match_row[0]
            result_dict[match_row] = data
        else:
            # Miss: yield the match_row for the caller to compute/insert
            yield dict(zip(match_cols.keys(), match_row))


def insert_via_arrow_limitoffset(conn, table, *, chunk_size=None, **kwargs):
    import pyarrow as pa

    arrow_table = pa.table(dict(kwargs))

    conn.register("arrow_table", arrow_table)
    try:
        offset = 0
        while True:
            try:
                if chunk_size is not None:
                    while offset < arrow_table.num_rows:
                        conn.execute(
                            f"insert into {table} by name select * from arrow_table limit {chunk_size} offset {offset}"
                        )
                        offset += chunk_size
                else:
                    conn.execute(
                        f"insert into {table} by name select * from arrow_table"
                    )
            except duckdb.OutOfMemoryException:
                if chunk_size is None:
                    chunk_size = arrow_table.num_rows
                if chunk_size <= 1:
                    raise
                chunk_size = chunk_size // 2
                print(f"Out of memory, reducing chunk size to {chunk_size}")
            else:
                break
    finally:
        conn.unregister("arrow_table")


def insert_via_arrow_recordbatches(conn, table, *, chunk_size=None, **kwargs):
    import pyarrow as pa

    tables = [pa.table(dict(kwargs))]

    while len(tables) > 0:
        current_table = tables[0]
        conn.register("arrow_table", current_table)
        try:
            conn.execute(f"insert into {table} by name select * from arrow_table")
        except duckdb.OutOfMemoryException:
            if chunk_size is None:
                chunk_size = current_table.num_rows
            chunk_size = chunk_size // 2
            if chunk_size <= 1:
                raise
            print(f"Out of memory, reducing chunk size to {chunk_size}")
            conn.unregister("arrow_table")
            new_tables = []
            for table in tables:
                for chunk in table.to_batches(chunk_size):
                    new_tables.append(pa.Table.from_batches([chunk]))
            tables = new_tables
        else:
            tables.pop(0)
        finally:
            conn.unregister("arrow_table")


insert_via_arrow = insert_via_arrow_recordbatches
