"""Buffer-protocol (numpy) input and add_batch — issues #19 and #20.

CI excludes vanedb-py; these run locally via maturin develop + pytest.
"""
import pytest

np = pytest.importorskip("numpy")
import vanedb


# --- single-vector buffer input (issue #20) ---

def test_store_add_and_search_numpy_f32():
    store = vanedb.FlatIndex(3)
    store.add(1, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    store.add(2, np.array([4.0, 5.0, 6.0], dtype=np.float32))
    assert store.get(1) == [1.0, 2.0, 3.0]
    results = store.search(np.array([1.0, 2.0, 3.1], dtype=np.float32), 1)
    assert results[0][0] == 1


def test_store_add_numpy_f64_falls_back():
    # float64 has no f32 fast path but must still work via the sequence path
    store = vanedb.FlatIndex(2)
    store.add(1, np.array([1.0, 2.0]))  # default dtype float64
    assert store.get(1) == [1.0, 2.0]


def test_store_add_2d_buffer_rejected_for_single_add():
    store = vanedb.FlatIndex(2)
    with pytest.raises(ValueError, match="1-D"):
        store.add(1, np.zeros((2, 2), dtype=np.float32))


def test_lists_still_work():
    store = vanedb.FlatIndex(2)
    store.add(1, [1.0, 2.0])
    assert store.search([1.0, 2.0], 1)[0][0] == 1


# --- add_batch (issue #19) ---

def test_store_add_batch_numpy():
    rng = np.random.default_rng(0)
    vecs = rng.random((100, 8), dtype=np.float32)
    ids = np.arange(100, dtype=np.uint64)
    store = vanedb.FlatIndex(8)
    store.add_batch(ids, vecs)
    assert len(store) == 100
    results = store.search(vecs[42], 1)
    assert results[0][0] == 42


def test_store_add_batch_int64_ids():
    # np.arange default dtype is int64; must be accepted
    store = vanedb.FlatIndex(2)
    store.add_batch(np.arange(3), np.ones((3, 2), dtype=np.float32))
    assert len(store) == 3


def test_store_add_batch_negative_id_raises():
    store = vanedb.FlatIndex(2)
    with pytest.raises(ValueError, match="negative"):
        store.add_batch(np.array([0, -1]), np.ones((2, 2), dtype=np.float32))
    assert len(store) == 0


def test_store_add_batch_list_fallback():
    store = vanedb.FlatIndex(2)
    store.add_batch([10, 20], [[1.0, 2.0], [3.0, 4.0]])
    assert store.get(20) == [3.0, 4.0]


def test_store_add_batch_ragged_rows_raise():
    store = vanedb.FlatIndex(3)
    # total length 6 == 2*3 would pass a naive flat check; per-row must fail
    with pytest.raises(ValueError):
        store.add_batch([1, 2], [[1.0, 2.0], [3.0, 4.0, 5.0, 6.0]])
    assert len(store) == 0


def test_store_add_batch_wrong_width_raises():
    store = vanedb.FlatIndex(3)
    with pytest.raises(ValueError):
        store.add_batch(np.arange(2), np.ones((2, 4), dtype=np.float32))
    assert len(store) == 0


def test_store_add_batch_ids_vectors_count_mismatch():
    store = vanedb.FlatIndex(2)
    with pytest.raises(ValueError):
        store.add_batch(np.arange(3), np.ones((2, 2), dtype=np.float32))
    assert len(store) == 0


def test_store_add_batch_numpy_f64_2d():
    # float64 is numpy's default dtype; a well-formed 2-D batch must insert
    store = vanedb.FlatIndex(3)
    store.add_batch(np.arange(2), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
    assert len(store) == 2
    assert store.get(1) == [4.0, 5.0, 6.0]


def test_store_add_batch_f64_wrong_rank_raises():
    # Shape validation must not depend on dtype: the float32 twin of this
    # array is rejected as 3-D, so the float64 one must be too — not silently
    # flattened through numpy's element coercion.
    store = vanedb.FlatIndex(3)
    with pytest.raises(ValueError, match="2-D"):
        store.add_batch(np.arange(2), np.zeros((2, 3, 1)))
    assert len(store) == 0


def test_store_add_f64_wrong_rank_raises():
    # float64 twin of test_store_add_2d_buffer_rejected_for_single_add
    store = vanedb.FlatIndex(3)
    with pytest.raises(ValueError, match="1-D"):
        store.add(1, np.zeros((3, 1)))
    assert len(store) == 0


def test_store_add_batch_duplicate_is_all_or_nothing():
    store = vanedb.FlatIndex(2)
    store.add(5, [0.0, 0.0])
    with pytest.raises(ValueError, match="duplicate"):
        store.add_batch(np.array([4, 5], dtype=np.uint64),
                        np.ones((2, 2), dtype=np.float32))
    assert len(store) == 1
    assert not store.contains(4)


def test_store_add_batch_noncontiguous_slice():
    vecs = np.arange(40, dtype=np.float32).reshape(10, 4)
    view = vecs[::2]  # non C-contiguous
    store = vanedb.FlatIndex(4)
    store.add_batch(np.arange(5, dtype=np.uint64), view)
    assert store.get(1) == view[1].tolist()


# --- HNSW ---

def test_hnsw_add_batch_matches_serial():
    rng = np.random.default_rng(1)
    vecs = rng.random((200, 16), dtype=np.float32)
    ids = np.arange(200, dtype=np.uint64)

    serial = vanedb.ApproxIndex(16, capacity=200, seed=3)
    for i in range(200):
        serial.add(int(ids[i]), vecs[i])
    batched = vanedb.ApproxIndex(16, capacity=200, seed=3)
    batched.add_batch(ids, vecs)

    assert len(batched) == 200
    q = rng.random(16, dtype=np.float32)
    assert serial.search(q, 10) == batched.search(q, 10)


def test_hnsw_add_batch_grows_past_the_capacity_hint():
    """A batch running past the hint grows storage rather than failing.

    All-or-nothing on a genuine failure is covered by the duplicate-id case.
    """
    index = vanedb.ApproxIndex(2, capacity=3)
    index.add_batch(np.arange(8), np.ones((8, 2), dtype=np.float32))
    assert len(index) == 8


# --- The corpus must survive an awkward memory layout ---


def _numpy_ranking(vectors, ids, query, k):
    """Ground truth in float64, ties by ascending id, computed here."""
    v = np.asarray(vectors, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    d = ((v - q) ** 2).sum(axis=1)
    order = sorted(range(len(ids)), key=lambda i: (d[i], ids[i]))
    return [ids[i] for i in order[:k]]


@pytest.mark.parametrize(
    "layout",
    ["c_contiguous", "fortran", "transposed", "column_strided", "reversed_rows"],
)
def test_search_ranking_survives_every_buffer_layout(layout):
    """A batch read at the wrong stride stores a transposed corpus, and every
    later search silently answers from the wrong vectors.

    `test_store_add_batch_noncontiguous_slice` above takes `vecs[::2]`, which
    is row-strided and therefore still has each row contiguous — the layout
    that cannot expose a transposition. These do: Fortran order and a
    transposed view are column-major, so reading them as C-contiguous returns a
    different corpus that is the same size and full of the same numbers.

    The judge is numpy in float64, not another vanedb index, so the check does
    not depend on the extraction path it is testing.
    """
    rng = np.random.default_rng(90909)
    base = rng.standard_normal((40, 6)).astype(np.float32)
    if layout == "c_contiguous":
        vecs = np.ascontiguousarray(base)
    elif layout == "fortran":
        vecs = np.asfortranarray(base)
    elif layout == "transposed":
        vecs = rng.standard_normal((6, 40)).astype(np.float32).T
    elif layout == "column_strided":
        vecs = rng.standard_normal((40, 12)).astype(np.float32)[:, ::2]
    else:
        vecs = np.ascontiguousarray(base)[::-1]
    assert vecs.shape == (40, 6)

    ids = [i * 3 + 1 for i in range(40)]
    store = vanedb.FlatIndex(6, vanedb.Metric.L2)
    store.add_batch(ids, vecs)

    # Every stored vector is the row the caller passed, not a column of it.
    for i, want in zip(ids, vecs):
        assert store.get(i) == pytest.approx(want.tolist()), f"{layout}: row {i}"

    query = rng.standard_normal(6).astype(np.float32)
    got = [i for i, _ in store.search(query, 10)]
    assert got == _numpy_ranking(vecs, ids, query, 10), layout


def test_a_transposed_batch_is_not_silently_accepted_at_the_wrong_shape():
    """The shape check must read the buffer's own shape, not infer one from
    the element count. A (6, 40) array holds exactly as many floats as the
    (40, 6) the index wants."""
    wrong = np.zeros((6, 40), dtype=np.float32)
    store = vanedb.FlatIndex(6, vanedb.Metric.L2)
    with pytest.raises(ValueError):
        store.add_batch(list(range(6)), wrong)
    assert len(store) == 0
