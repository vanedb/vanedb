import vanedb
import os
import tempfile


def test_version():
    assert vanedb.__version__ == "0.1.0"


# --- VectorStore ---

def test_vector_store_basic():
    store = vanedb.PyVectorStore(3)
    store.add(1, [1.0, 2.0, 3.0])
    store.add(2, [4.0, 5.0, 6.0])
    assert len(store) == 2
    assert store.dimension == 3
    assert store.contains(1)
    assert not store.contains(99)


def test_vector_store_get():
    store = vanedb.PyVectorStore(3)
    store.add(1, [1.0, 2.0, 3.0])
    assert store.get(1) == [1.0, 2.0, 3.0]


def test_vector_store_search():
    store = vanedb.PyVectorStore(2)
    store.add(1, [0.0, 0.0])
    store.add(2, [1.0, 0.0])
    store.add(3, [10.0, 10.0])
    results = store.search([0.0, 0.1], 2)
    assert len(results) == 2
    assert results[0][0] == 1  # closest


def test_vector_store_cosine():
    store = vanedb.PyVectorStore(2, vanedb.PyDistanceMetric.Cosine)
    store.add(1, [1.0, 0.0])
    store.add(2, [0.0, 1.0])
    results = store.search([0.9, 0.1], 1)
    assert results[0][0] == 1


def test_vector_store_remove():
    store = vanedb.PyVectorStore(2)
    store.add(1, [1.0, 2.0])
    store.add(2, [3.0, 4.0])
    store.remove(1)
    assert len(store) == 1
    assert not store.contains(1)
    assert store.contains(2)


def test_vector_store_errors():
    store = vanedb.PyVectorStore(3)
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


# --- HnswIndex ---

def test_hnsw_basic():
    idx = vanedb.PyHnswIndex(3, capacity=100)
    idx.add(1, [1.0, 0.0, 0.0])
    idx.add(2, [0.0, 1.0, 0.0])
    assert len(idx) == 2
    assert idx.dimension == 3
    assert idx.capacity == 100
    assert idx.contains(1)


def test_hnsw_search():
    idx = vanedb.PyHnswIndex(3, capacity=100)
    idx.add(1, [0.0, 0.0, 0.0])
    idx.add(2, [10.0, 10.0, 10.0])
    results = idx.search([0.0, 0.0, 0.0], 1)
    assert results[0][0] == 1
    assert results[0][1] < 1e-6  # exact match


def test_hnsw_save_load():
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name

    try:
        idx = vanedb.PyHnswIndex(4, capacity=100, seed=42)
        for i in range(20):
            idx.add(i, [float(i)] * 4)
        idx.save(path)

        loaded = vanedb.PyHnswIndex.load(path)
        assert len(loaded) == 20
        assert loaded.get_vector(5) == [5.0, 5.0, 5.0, 5.0]

        # Search results should match
        r1 = idx.search([5.5] * 4, 3)
        r2 = loaded.search([5.5] * 4, 3)
        assert [r[0] for r in r1] == [r[0] for r in r2]
    finally:
        os.unlink(path)


def test_hnsw_ef_search():
    idx = vanedb.PyHnswIndex(3, capacity=100)
    assert idx.ef_search == 50  # default
    idx.ef_search = 200
    assert idx.ef_search == 200


def test_hnsw_errors():
    idx = vanedb.PyHnswIndex(3, capacity=2)
    idx.add(0, [0.0, 0.0, 0.0])
    idx.add(1, [1.0, 1.0, 1.0])
    try:
        idx.add(2, [2.0, 2.0, 2.0])  # full
        assert False, "Should have raised"
    except ValueError:
        pass
