"""Buffer-protocol (numpy) input and add_batch — issues #19 and #20.

CI excludes vanedb-py; these run locally via maturin develop + pytest.
"""
import pytest

np = pytest.importorskip("numpy")
import vanedb


# --- single-vector buffer input (issue #20) ---

def test_store_add_and_search_numpy_f32():
    store = vanedb.PyVectorStore(3)
    store.add(1, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    store.add(2, np.array([4.0, 5.0, 6.0], dtype=np.float32))
    assert store.get(1) == [1.0, 2.0, 3.0]
    results = store.search(np.array([1.0, 2.0, 3.1], dtype=np.float32), 1)
    assert results[0][0] == 1


def test_store_add_numpy_f64_falls_back():
    # float64 has no f32 fast path but must still work via the sequence path
    store = vanedb.PyVectorStore(2)
    store.add(1, np.array([1.0, 2.0]))  # default dtype float64
    assert store.get(1) == [1.0, 2.0]


def test_store_add_2d_buffer_rejected_for_single_add():
    store = vanedb.PyVectorStore(2)
    with pytest.raises(ValueError, match="1-D"):
        store.add(1, np.zeros((2, 2), dtype=np.float32))


def test_lists_still_work():
    store = vanedb.PyVectorStore(2)
    store.add(1, [1.0, 2.0])
    assert store.search([1.0, 2.0], 1)[0][0] == 1


# --- add_batch (issue #19) ---

def test_store_add_batch_numpy():
    rng = np.random.default_rng(0)
    vecs = rng.random((100, 8), dtype=np.float32)
    ids = np.arange(100, dtype=np.uint64)
    store = vanedb.PyVectorStore(8)
    store.add_batch(ids, vecs)
    assert len(store) == 100
    results = store.search(vecs[42], 1)
    assert results[0][0] == 42


def test_store_add_batch_int64_ids():
    # np.arange default dtype is int64; must be accepted
    store = vanedb.PyVectorStore(2)
    store.add_batch(np.arange(3), np.ones((3, 2), dtype=np.float32))
    assert len(store) == 3


def test_store_add_batch_negative_id_raises():
    store = vanedb.PyVectorStore(2)
    with pytest.raises(ValueError, match="negative"):
        store.add_batch(np.array([0, -1]), np.ones((2, 2), dtype=np.float32))
    assert len(store) == 0


def test_store_add_batch_list_fallback():
    store = vanedb.PyVectorStore(2)
    store.add_batch([10, 20], [[1.0, 2.0], [3.0, 4.0]])
    assert store.get(20) == [3.0, 4.0]


def test_store_add_batch_ragged_rows_raise():
    store = vanedb.PyVectorStore(3)
    # total length 6 == 2*3 would pass a naive flat check; per-row must fail
    with pytest.raises(ValueError):
        store.add_batch([1, 2], [[1.0, 2.0], [3.0, 4.0, 5.0, 6.0]])
    assert len(store) == 0


def test_store_add_batch_wrong_width_raises():
    store = vanedb.PyVectorStore(3)
    with pytest.raises(ValueError):
        store.add_batch(np.arange(2), np.ones((2, 4), dtype=np.float32))
    assert len(store) == 0


def test_store_add_batch_ids_vectors_count_mismatch():
    store = vanedb.PyVectorStore(2)
    with pytest.raises(ValueError):
        store.add_batch(np.arange(3), np.ones((2, 2), dtype=np.float32))
    assert len(store) == 0


def test_store_add_batch_duplicate_is_all_or_nothing():
    store = vanedb.PyVectorStore(2)
    store.add(5, [0.0, 0.0])
    with pytest.raises(ValueError, match="duplicate"):
        store.add_batch(np.array([4, 5], dtype=np.uint64),
                        np.ones((2, 2), dtype=np.float32))
    assert len(store) == 1
    assert not store.contains(4)


def test_store_add_batch_noncontiguous_slice():
    vecs = np.arange(40, dtype=np.float32).reshape(10, 4)
    view = vecs[::2]  # non C-contiguous
    store = vanedb.PyVectorStore(4)
    store.add_batch(np.arange(5, dtype=np.uint64), view)
    assert store.get(1) == view[1].tolist()


# --- HNSW ---

def test_hnsw_add_batch_matches_serial():
    rng = np.random.default_rng(1)
    vecs = rng.random((200, 16), dtype=np.float32)
    ids = np.arange(200, dtype=np.uint64)

    serial = vanedb.PyHnswIndex(16, capacity=200, seed=3)
    for i in range(200):
        serial.add(int(ids[i]), vecs[i])
    batched = vanedb.PyHnswIndex(16, capacity=200, seed=3)
    batched.add_batch(ids, vecs)

    assert len(batched) == 200
    q = rng.random(16, dtype=np.float32)
    assert serial.search(q, 10) == batched.search(q, 10)


def test_hnsw_add_batch_capacity_all_or_nothing():
    index = vanedb.PyHnswIndex(2, capacity=3)
    with pytest.raises(ValueError, match="full"):
        index.add_batch(np.arange(4), np.ones((4, 2), dtype=np.float32))
    assert len(index) == 0
