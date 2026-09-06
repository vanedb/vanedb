import vanedb
import os
import tempfile

import pytest


def test_version():
    # Compared against the distribution's own metadata rather than a literal:
    # a literal has to be edited on every bump, and silently passed while
    # __version__ and the wheel metadata disagreed. Version() normalises, so
    # the SemVer spelling (0.1.0-rc.1) and the PEP 440 one (0.1.0rc1) match.
    from importlib.metadata import version

    from packaging.version import Version

    assert Version(vanedb.__version__) == Version(version("vanedb"))


# --- FlatIndex ---

def test_vector_store_basic():
    store = vanedb.FlatIndex(3)
    store.add(1, [1.0, 2.0, 3.0])
    store.add(2, [4.0, 5.0, 6.0])
    assert len(store) == 2
    assert store.dimension == 3
    assert store.contains(1)
    assert not store.contains(99)


def test_vector_store_get():
    store = vanedb.FlatIndex(3)
    store.add(1, [1.0, 2.0, 3.0])
    assert store.get(1) == [1.0, 2.0, 3.0]


def test_vector_store_search():
    store = vanedb.FlatIndex(2)
    store.add(1, [0.0, 0.0])
    store.add(2, [1.0, 0.0])
    store.add(3, [10.0, 10.0])
    results = store.search([0.0, 0.1], 2)
    assert len(results) == 2
    assert results[0][0] == 1  # closest


def test_vector_store_cosine():
    store = vanedb.FlatIndex(2, vanedb.Metric.COSINE)
    store.add(1, [1.0, 0.0])
    store.add(2, [0.0, 1.0])
    results = store.search([0.9, 0.1], 1)
    assert results[0][0] == 1


def test_vector_store_remove():
    store = vanedb.FlatIndex(2)
    store.add(1, [1.0, 2.0])
    store.add(2, [3.0, 4.0])
    store.remove(1)
    assert len(store) == 1
    assert not store.contains(1)
    assert store.contains(2)


def test_vector_store_errors():
    store = vanedb.FlatIndex(3)
    try:
        store.add(1, [1.0, 2.0])  # wrong dim
        assert False, "Should have raised"
    except ValueError:
        pass

    store.add(1, [1.0, 2.0, 3.0])
    try:
        store.add(1, [4.0, 5.0, 6.0])  # duplicate
        assert False, "Should have raised"
    except ValueError:
        pass


# --- ApproxIndex ---

def test_hnsw_basic():
    idx = vanedb.ApproxIndex(3, capacity=100)
    idx.add(1, [1.0, 0.0, 0.0])
    idx.add(2, [0.0, 1.0, 0.0])
    assert len(idx) == 2
    assert idx.dimension == 3
    assert idx.capacity == 100
    assert idx.contains(1)


def test_hnsw_search():
    idx = vanedb.ApproxIndex(3, capacity=100)
    idx.add(1, [0.0, 0.0, 0.0])
    idx.add(2, [10.0, 10.0, 10.0])
    results = idx.search([0.0, 0.0, 0.0], 1)
    assert results[0][0] == 1
    assert results[0][1] < 1e-6  # exact match


def test_hnsw_save_load():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name

    try:
        idx = vanedb.ApproxIndex(4, capacity=100, seed=42)
        for i in range(20):
            idx.add(i, [float(i)] * 4)
        idx.save(path)

        loaded = vanedb.ApproxIndex.load(path)
        assert len(loaded) == 20
        assert loaded.get_vector(5) == [5.0, 5.0, 5.0, 5.0]

        # Search results should match
        r1 = idx.search([5.5] * 4, 3)
        r2 = loaded.search([5.5] * 4, 3)
        assert [r[0] for r in r1] == [r[0] for r in r2]
    finally:
        os.unlink(path)


def test_hnsw_ef_search():
    idx = vanedb.ApproxIndex(3, capacity=100)
    assert idx.ef_search == 50  # default
    idx.ef_search = 200
    assert idx.ef_search == 200


def test_hnsw_grows_past_the_capacity_hint():
    """capacity reserves; it does not cap."""
    idx = vanedb.ApproxIndex(3, capacity=2)
    for i in range(20):
        idx.add(i, [float(i)] * 3)
    assert len(idx) == 20
    assert idx.search([19.0, 19.0, 19.0], 1)[0][0] == 19


def test_hnsw_errors():
    idx = vanedb.ApproxIndex(3, capacity=2)
    idx.add(0, [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        idx.add(0, [1.0, 1.0, 1.0])  # duplicate id
    with pytest.raises(ValueError):
        idx.add(1, [1.0, 1.0])  # wrong dimension


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_vectors_and_queries_are_rejected(value):
    store = vanedb.FlatIndex(2)
    with pytest.raises(ValueError, match="finite"):
        store.add(1, [value, 0.0])
    assert len(store) == 0

    store.add(2, [0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        store.search([value, 0.0], 1)

    index = vanedb.ApproxIndex(2, capacity=4)
    with pytest.raises(ValueError, match="finite"):
        index.add(1, [value, 0.0])
    assert len(index) == 0


def test_count_spellings_agree():
    """Both engines must answer "how many vectors?" the same way (#85)."""
    store = vanedb.FlatIndex(2)
    index = vanedb.ApproxIndex(2, capacity=10)
    for i, v in enumerate([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]):
        store.add(i, v)
        index.add(i, v)
    assert len(store) == store.size() == 3
    assert len(index) == index.size() == 3


def test_public_surface_is_declared():
    """__all__ is the whole package surface, not just a star-import filter.

    maturin's generated __init__ copies __all__ verbatim after a star import,
    so a name missing here is missing from the package: dropping __version__
    from this list deleted vanedb.__version__ outright.
    """
    assert set(vanedb.__all__) == {
        "Metric",
        "FlatIndex",
        "ApproxIndex",
        "DiskIndex",
        "DiskIndexBuilder",
        "__version__",
    }
    for name in vanedb.__all__:
        assert hasattr(vanedb, name), f"{name} is exported but missing"


# --- DiskIndex ---


def test_disk_store_round_trip(tmp_path):
    """The memory-mapped store must be reachable from Python at all (#84)."""
    path = str(tmp_path / "store.vndb")
    builder = vanedb.DiskIndexBuilder(3, vanedb.Metric.L2)
    for i in range(20):
        builder.add(i, [float(i), 1.0, 2.0])
    assert len(builder) == 20
    assert builder.dimension == 3
    builder.save(path)

    store = vanedb.DiskIndex.open(path)
    assert len(store) == store.size() == 20
    assert store.dimension == 3
    assert store.contains(7)
    assert not store.contains(999)
    assert store.get(7) == [7.0, 1.0, 2.0]
    hits = store.search([7.0, 1.0, 2.0], 1)
    assert hits[0][0] == 7


def test_disk_store_rejects_a_bad_file(tmp_path):
    path = tmp_path / "not-a-store.vndb"
    path.write_bytes(b"nonsense" * 8)
    with pytest.raises(ValueError):
        vanedb.DiskIndex.open(str(path))


def test_disk_store_is_in_the_public_surface():
    assert "DiskIndex" in vanedb.__all__
    assert "DiskIndexBuilder" in vanedb.__all__


def test_approx_index_delete():
    """Deletion is what the release was waiting on (#91)."""
    idx = vanedb.ApproxIndex(2, capacity=64)
    for i in range(40):
        idx.add(i, [float(i), 0.0])
    assert len(idx) == 40

    idx.remove(7)
    assert len(idx) == 39
    assert not idx.contains(7)
    assert all(hit[0] != 7 for hit in idx.search([7.0, 0.0], 5))

    with pytest.raises(ValueError):
        idx.remove(7)

    # The id is free again.
    idx.add(7, [700.0, 0.0])
    assert len(idx) == 40
    assert idx.search([700.0, 0.0], 1)[0][0] == 7
