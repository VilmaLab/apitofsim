import mmap

import numpy as np
import pytest
from apitofsim.workflow import DatabasePaths, HistogramStore, SuperClusterDatabase


def test_native_rows_and_mapping_lifetime(tmp_path):
    paths = DatabasePaths(tmp_path / "clusters.apitofsim")
    paths.create()
    writer = HistogramStore(paths)
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    assert list(writer.append(values[:2])) == [0, 1]
    assert list(writer.append(values[0, :2][None, :])) == [0]
    if writer._use_mmap:
        reader = writer
    else:
        # Windows can map the immutable files once writing is finished.
        assert list(writer.append(values[2:])) == [2]
        reader = HistogramStore(paths, readonly=True)
    reader.premap()
    first, second = reader.read_rows(4, (0, 1))
    assert first.base.obj is second.base.obj
    assert isinstance(first.base.obj, mmap.mmap)
    assert not first.flags.writeable
    if reader is writer:
        assert list(writer.append(values[2:])) == [2]
    (third,) = reader.read_rows(4, (2,))
    reader.release_mappings()
    np.testing.assert_array_equal(first, values[0])
    np.testing.assert_array_equal(third, values[2])
    assert paths.histograms(4).read_bytes() == values.tobytes()
    with pytest.raises(ValueError, match="exceeds the size"):
        reader.read_rows(4, (3,))
    with pytest.raises(FileNotFoundError):
        reader.read_rows(5, (0,))


def test_rolled_back_addresses_do_not_publish(tmp_path):
    path = tmp_path / "clusters.apitofsim"
    with SuperClusterDatabase(path) as db:
        db.create_tables()
        values = np.array([[1.0, 2.0, 3.0]])
        db.db.execute("begin transaction")
        db.insert_histograms("cluster_dos", 1, values, cluster_id=[1])
        db.db.execute("rollback")
        db.insert_histograms(
            "products_dos", 1, values * 2, cluster1_id=[1], cluster2_id=[2]
        )
        assert db.db.sql("select * from cluster_dos").fetchall() == []
        assert db.db.sql("select histogram_row from products_dos").fetchone() == (1,)
    with SuperClusterDatabase(path, readonly=True) as db:
        (stored,) = db.histogram_store.read_rows(3, (1,))
        np.testing.assert_array_equal(stored, values[0] * 2)
        with pytest.raises(PermissionError, match="read-only"):
            db.insert_histograms("cluster_dos", 1, values, cluster_id=[1])
