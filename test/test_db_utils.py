from pathlib import Path

import duckdb
import pytest
from apitofsim.workflow.db_utils import duckdb_connect_roview_cow


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "test.duckdb"
    conn = duckdb.connect(str(path))
    conn.execute("create table t as select 42 as x")
    conn.close()
    return path


@pytest.mark.parametrize("as_path", [False, True])
def test_roview_cow_accepts_pathlike(database, as_path):
    filename = database if as_path else str(database)
    db, cleanup = duckdb_connect_roview_cow(filename, fallback="connect")
    try:
        assert db.execute("select x from t").fetchone() == (42,)
    finally:
        db.close()
        if cleanup is not None:
            cleanup()


def test_roview_cow_gives_reflink_strings(database, monkeypatch):
    reflink = pytest.importorskip("reflink")
    calls = []

    def fake_reflink(oldpath, newpath):
        calls.append((oldpath, newpath))
        raise reflink.ReflinkImpossibleError("pretend this filesystem cannot")

    monkeypatch.setattr(reflink, "reflink", fake_reflink)
    db, _ = duckdb_connect_roview_cow(Path(database), fallback="connect")
    try:
        assert [type(arg) for call in calls for arg in call] == [str, str]
    finally:
        db.close()


def test_cached_arrays_follow_requested_cluster_ids():
    from apitofsim.workflow.db_utils import get_through_join_else

    with duckdb.connect() as conn:
        conn.execute(
            "create table curves as select i as id, [i::double] as data from range(40) a(i)"
        )
        # Older caches may contain duplicate rows for a shared pair of products.
        conn.execute("insert into curves select * from curves where id = 1")
        cached = {}
        missing = list(
            get_through_join_else(
                conn, conn.table("curves"), "data", cached, id=list(reversed(range(41)))
            )
        )
        assert missing == [{"id": 40}]
        assert all(value.tolist() == [key] for key, value in cached.items())
