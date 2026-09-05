"""
Python binding tests for VaneDB Index.

Run with: uv run pytest tests/test_python_bindings.py -v
"""

import pytest
import numpy as np
import tempfile
import os
import gc
import subprocess
import sys
import textwrap
import weakref


def test_import():
    """Test that the module can be imported."""
    import vanedb_cpp
    assert hasattr(vanedb_cpp, 'Index')


def test_version():
    """Test that version info is accessible."""
    import vanedb_cpp
    assert hasattr(vanedb_cpp, '__version__')
    assert isinstance(vanedb_cpp.__version__, str)
    from importlib.metadata import version

    from packaging.version import Version

    # Against the distribution metadata, not a literal — see the note in
    # vanedb-py/tests/test_vanedb.py.
    assert Version(vanedb_cpp.__version__) == Version(version("vanedb-cpp"))
    assert vanedb_cpp.VERSION_MAJOR == 0
    assert vanedb_cpp.VERSION_MINOR == 1
    assert vanedb_cpp.VERSION_PATCH == 0


def test_simd_backend():
    import vanedb_cpp
    backend = vanedb_cpp.simd_backend()
    assert backend in {"scalar", "neon", "avx2_fma"}
    # The QEMU acceptance job checks actual selection, not merely no crash.
    expected = os.environ.get("VANEDB_EXPECT_BACKEND")
    if expected:
        assert backend == expected


@pytest.mark.parametrize("dimension", [7, 8, 31, 32, 33, 128, 773])
@pytest.mark.parametrize("metric", ["L2", "COSINE", "DOT"])
def test_dispatched_search_matches_numpy(dimension, metric):
    import vanedb_cpp
    rng = np.random.default_rng(59)
    vectors = rng.normal(size=(24, dimension)).astype(np.float32)
    query = rng.normal(size=dimension).astype(np.float32)
    index = vanedb_cpp.Index(
        dimension, getattr(vanedb_cpp.Metric, metric), max_elements=32
    )
    index.set_ef_search(64)
    for i, vector in enumerate(vectors):
        index.add(i + 1000, vector)
    data, q = vectors.astype(np.float64), query.astype(np.float64)
    # Avoid BLAS here: its vendor/model dispatch can assume FMA on a Haswell
    # whose FMA flag a VM deliberately masks. This oracle tests our kernels,
    # not OpenBLAS's separate CPU dispatch policy.
    dot = np.sum(data * q, axis=1)
    if metric == "L2":
        distances = np.sum((data - q) ** 2, axis=1)
    elif metric == "DOT":
        distances = -dot
    else:
        distances = 1 - dot / (np.sqrt(np.sum(data * data, axis=1)) * np.sqrt(np.sum(q * q)))
    expected = np.argsort(distances)[:5]
    ids, actual = index.search(query, 5)
    np.testing.assert_array_equal(ids, expected + 1000)
    np.testing.assert_allclose(actual, distances[expected], rtol=2e-5, atol=2e-5)


def test_distance_metrics():
    """Test that distance metric enum values are accessible."""
    import vanedb_cpp
    assert vanedb_cpp.Metric.L2 is not None
    assert vanedb_cpp.Metric.COSINE is not None
    assert vanedb_cpp.Metric.DOT is not None


def test_create_index_default():
    """Test creating an index with default parameters."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=128)
    assert index.size() == 0
    assert index.dimension() == 128
    assert index.capacity() == 100000


def test_create_index_custom():
    """Test creating an index with custom parameters."""
    import vanedb_cpp
    index = vanedb_cpp.Index(
        dimension=64,
        metric=vanedb_cpp.Metric.COSINE,
        max_elements=1000,
        M=32,
        ef_construction=100,
        random_seed=123
    )
    assert index.size() == 0
    assert index.dimension() == 64
    assert index.capacity() == 1000


def test_add_single_vector():
    """Test adding a single vector."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add(1, vec)

    assert index.size() == 1
    assert index.contains(1)
    assert not index.contains(2)


def test_add_multiple_vectors():
    """Test adding multiple vectors."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    for i in range(100):
        vec = np.random.randn(4).astype(np.float32)
        index.add(i, vec)

    assert index.size() == 100
    for i in range(100):
        assert index.contains(i)


def test_search_basic():
    """Test basic search functionality."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add(42, vec)

    ids, dists = index.search(vec, 1)
    assert len(ids) == 1
    assert len(dists) == 1
    assert ids[0] == 42
    assert dists[0] < 1e-6  # Should be ~0 for exact match


def test_search_knn():
    """Test k-nearest neighbor search."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    # Add 10 vectors
    for i in range(10):
        vec = np.array([float(i), 0.0, 0.0, 0.0], dtype=np.float32)
        index.add(i, vec)

    # Search for 5 nearest to [5, 0, 0, 0]
    query = np.array([5.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = index.search(query, 5)

    assert len(ids) == 5
    assert len(dists) == 5
    assert ids[0] == 5  # Exact match should be first
    assert dists[0] < 1e-6


def test_search_returns_numpy_arrays():
    """Test that search returns numpy arrays."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add(1, vec)

    ids, dists = index.search(vec, 1)
    assert isinstance(ids, np.ndarray)
    assert isinstance(dists, np.ndarray)
    assert ids.dtype == np.uint64
    assert dists.dtype == np.float32


def test_get_vector():
    """Test retrieving a stored vector."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    index.add(42, original)

    retrieved = index.get_vector(42)
    assert isinstance(retrieved, np.ndarray)
    assert retrieved.dtype == np.float32
    assert len(retrieved) == 4
    np.testing.assert_array_almost_equal(retrieved, original)


def test_ef_search():
    """Test setting and getting ef_search parameter."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    # Default should be reasonable
    default_ef = index.get_ef_search()
    assert default_ef > 0

    # Set new value
    index.set_ef_search(100)
    assert index.get_ef_search() == 100


def test_save_load(tmp_path):
    """Test saving and loading an index."""
    import vanedb_cpp

    # Create and populate index
    index = vanedb_cpp.Index(dimension=4)
    vectors = {}
    for i in range(10):
        vec = np.random.randn(4).astype(np.float32)
        vectors[i] = vec
        index.add(i, vec)

    # Save to file
    filepath = str(tmp_path / "test_index.bin")
    index.save(filepath)
    assert os.path.exists(filepath)

    # Load from file
    loaded = vanedb_cpp.Index.load(filepath)

    # Verify loaded index
    assert loaded.size() == 10
    assert loaded.dimension() == 4

    for i in range(10):
        assert loaded.contains(i)
        retrieved = loaded.get_vector(i)
        np.testing.assert_array_almost_equal(retrieved, vectors[i])


def test_save_load_search_consistency(tmp_path):
    """Test that search results are consistent after save/load."""
    import vanedb_cpp

    # Create and populate index
    index = vanedb_cpp.Index(dimension=8)
    np.random.seed(42)
    for i in range(100):
        vec = np.random.randn(8).astype(np.float32)
        index.add(i, vec)

    # Search before save
    query = np.random.randn(8).astype(np.float32)
    ids_before, dists_before = index.search(query, 10)

    # Save and load
    filepath = str(tmp_path / "test_index.bin")
    index.save(filepath)
    loaded = vanedb_cpp.Index.load(filepath)

    # Search after load
    ids_after, dists_after = loaded.search(query, 10)

    # Results should match
    np.testing.assert_array_equal(ids_before, ids_after)
    np.testing.assert_array_almost_equal(dists_before, dists_after)


def test_dimension_mismatch_add():
    """Test that adding wrong dimension vector raises error."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    wrong_dim = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 3 instead of 4
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        index.add(1, wrong_dim)


def test_dimension_mismatch_search():
    """Test that searching with wrong dimension query raises error."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    index.add(1, vec)

    wrong_query = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 3 instead of 4
    with pytest.raises(RuntimeError, match="dimension mismatch"):
        index.search(wrong_query, 1)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_vectors_and_queries_are_rejected(value):
    import vanedb_cpp

    invalid = np.array([value, 0.0], dtype=np.float32)
    finite = np.array([0.0, 0.0], dtype=np.float32)

    store = vanedb_cpp.Store(dimension=2)
    with pytest.raises(ValueError, match="finite"):
        store.add(1, invalid)
    assert store.size() == 0
    store.add(2, finite)
    with pytest.raises(ValueError, match="finite"):
        store.search(invalid, 1)

    index = vanedb_cpp.Index(dimension=2, max_elements=4)
    with pytest.raises(ValueError, match="finite"):
        index.add(1, invalid)
    assert index.size() == 0


def test_2d_array_add_raises():
    """Test that adding a 2D array raises error."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=4)

    vec_2d = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(RuntimeError, match="1-dimensional"):
        index.add(1, vec_2d)


def test_cosine_metric():
    """Test COSINE distance metric."""
    import vanedb_cpp
    index = vanedb_cpp.Index(
        dimension=4,
        metric=vanedb_cpp.Metric.COSINE
    )

    # Same direction vectors should have distance ~0
    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Same direction, different magnitude

    index.add(1, vec1)
    index.add(2, vec2)

    ids, dists = index.search(vec1, 2)
    # Both should have very small distances (cosine distance of parallel vectors)
    assert dists[0] < 0.01
    assert dists[1] < 0.01


def test_dot_metric():
    """Test DOT product metric."""
    import vanedb_cpp
    index = vanedb_cpp.Index(
        dimension=4,
        metric=vanedb_cpp.Metric.DOT
    )

    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)

    index.add(1, vec1)
    index.add(2, vec2)

    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = index.search(query, 2)

    # vec2 has larger dot product, should be first (negated in distance)
    assert ids[0] == 2


def test_large_scale():
    """Test with larger number of vectors."""
    import vanedb_cpp
    index = vanedb_cpp.Index(dimension=128, max_elements=10000)

    np.random.seed(42)
    for i in range(1000):
        vec = np.random.randn(128).astype(np.float32)
        index.add(i, vec)

    assert index.size() == 1000

    # Search should work
    query = np.random.randn(128).astype(np.float32)
    ids, dists = index.search(query, 10)

    assert len(ids) == 10
    assert len(dists) == 10
    # Results should be sorted by distance
    for i in range(len(dists) - 1):
        assert dists[i] <= dists[i + 1]


### Store Tests ###

def test_vector_store_import():
    """Test that Store can be imported."""
    import vanedb_cpp
    assert hasattr(vanedb_cpp, 'Store')


def test_vector_store_basic():
    """Test basic Store operations."""
    import vanedb_cpp
    store = vanedb_cpp.Store(dimension=4)

    vec = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    store.add(1, vec)

    assert store.size() == 1
    assert store.contains(1)

    retrieved = store.get(1)
    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, vec)


def test_vector_store_search():
    """Test Store search."""
    import vanedb_cpp
    store = vanedb_cpp.Store(dimension=4, metric=vanedb_cpp.Metric.L2)

    store.add(1, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    store.add(2, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))

    query = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = store.search(query, 1)

    assert ids[0] == 1


def test_vector_store_remove():
    """Test Store remove operation."""
    import vanedb_cpp
    store = vanedb_cpp.Store(dimension=4)

    store.add(1, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    assert store.size() == 1

    store.remove(1)
    assert store.size() == 0
    assert not store.contains(1)


### DiskStore Tests ###

def test_mmap_store_import():
    """Test that DiskStore can be imported."""
    import vanedb_cpp
    assert hasattr(vanedb_cpp, 'DiskStore')
    assert hasattr(vanedb_cpp, 'DiskStoreBuilder')


def test_mmap_store_build_and_load(tmp_path):
    """Test building and loading an mmap store."""
    import vanedb_cpp

    filepath = str(tmp_path / "test_mmap.bin")

    # Build
    builder = vanedb_cpp.DiskStoreBuilder(dimension=4)
    vec1 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vec2 = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    builder.add(10, vec1)
    builder.add(20, vec2)
    builder.save(filepath)

    # Load
    store = vanedb_cpp.DiskStore(filepath)
    assert store.size() == 2
    assert store.dimension() == 4
    assert store.contains(10)
    assert store.contains(20)


def test_mmap_store_search(tmp_path):
    """Test DiskStore search."""
    import vanedb_cpp

    filepath = str(tmp_path / "test_mmap_search.bin")

    builder = vanedb_cpp.DiskStoreBuilder(dimension=4)
    builder.add(1, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    builder.add(2, np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    builder.save(filepath)

    store = vanedb_cpp.DiskStore(filepath)
    query = np.array([0.9, 0.0, 0.0, 0.0], dtype=np.float32)
    ids, dists = store.search(query, 1)

    assert ids[0] == 1


def test_mmap_store_zero_copy_get(tmp_path):
    """Test that DiskStore get returns zero-copy array."""
    import vanedb_cpp

    filepath = str(tmp_path / "test_mmap_zerocopy.bin")

    builder = vanedb_cpp.DiskStoreBuilder(dimension=4)
    original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    builder.add(42, original)
    builder.save(filepath)

    store = vanedb_cpp.DiskStore(filepath)
    retrieved = store.get(42)

    assert retrieved is not None
    np.testing.assert_array_almost_equal(retrieved, original)
    assert retrieved.dtype == np.float32
    assert retrieved.shape == (4,)
    assert retrieved.strides == (original.itemsize,)
    assert not retrieved.flags.owndata
    assert retrieved.base is store
    assert np.shares_memory(retrieved, store.get(42))


def _mapped_vector(tmp_path):
    """Keep ownership in the caller so lifetime tests can release the store."""
    import vanedb_cpp

    path = tmp_path / "readonly.bin"
    original = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    builder = vanedb_cpp.DiskStoreBuilder(dimension=4)
    builder.add(42, original)
    builder.save(str(path))
    return path, original, vanedb_cpp.DiskStore(str(path))


@pytest.mark.parametrize("kind", ["direct", "slice", "asarray", "frombuffer"])
def test_mmap_views_are_readonly(tmp_path, kind):
    _, original, store = _mapped_vector(tmp_path)
    mapped = store.get(42)
    views = {
        "direct": mapped,
        "slice": mapped[::2],
        "asarray": np.asarray(mapped),
        "frombuffer": np.frombuffer(mapped, dtype=np.float32),
    }
    view = views[kind]
    assert np.shares_memory(view, mapped)
    assert not view.flags.writeable
    assert memoryview(view).readonly
    with pytest.raises(ValueError, match="WRITEABLE"):
        view.setflags(write=True)
    with pytest.raises(ValueError, match="WRITEABLE"):
        view.flags.writeable = True
    np.testing.assert_array_equal(store.get(42), original)


@pytest.mark.parametrize("kind", ["array", "slice", "memoryview"])
def test_mmap_views_keep_mapping_alive(tmp_path, kind):
    path, original, store = _mapped_vector(tmp_path)
    owner = weakref.ref(store)
    mapped = store.get(42)
    if kind == "slice":
        escaped = mapped[::2]
        expected = original[::2]
    elif kind == "memoryview":
        escaped = memoryview(mapped)
        expected = original
    else:
        escaped = mapped
        expected = original
    del mapped, store
    gc.collect()
    assert owner() is not None
    np.testing.assert_array_equal(escaped, expected)
    if kind == "memoryview":
        assert escaped.readonly
    else:
        assert not escaped.flags.writeable
    del escaped
    gc.collect()
    assert owner() is None
    # Windows also proves that the last view releases the mapped file handle.
    path.unlink()


def test_mmap_copy_is_independent_and_writable(tmp_path):
    path, original, store = _mapped_vector(tmp_path)
    before = path.read_bytes()
    mapped = store.get(42)
    editable = mapped.copy()
    assert editable.flags.writeable
    assert editable.flags.owndata
    assert not np.shares_memory(editable, mapped)
    editable[0] = 99.0
    np.testing.assert_array_equal(mapped, original)
    np.testing.assert_array_equal(store.get(42), original)
    ids, distances = store.search(original, 1)
    assert ids.tolist() == [42]
    assert distances.tolist() == [0.0]
    assert path.read_bytes() == before
    del mapped, store
    gc.collect()
    assert editable[0] == 99.0


def test_mmap_get_missing_id_returns_none(tmp_path):
    _, _, store = _mapped_vector(tmp_path)
    assert store.get(999) is None


@pytest.mark.parametrize("operation", [
    "direct", "slice", "memoryview", "ufunc_out", "copyto", "fill",
])
def test_mmap_writes_raise_without_crashing(tmp_path, operation):
    # Execute the actual write, without a preceding flags assertion. A future
    # regression must fail this test, not take down pytest with SIGSEGV/SIGBUS
    # (or a Windows access violation). Only a new temporary database is used.
    path, _, store = _mapped_vector(tmp_path)
    before = path.read_bytes()
    del store
    script = textwrap.dedent("""
        import os
        import sys
        if os.name == "nt":
            import ctypes
            # Suppress OS crash dialogs in the child, not the whole runner.
            ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        import numpy as np
        import vanedb_cpp
        store = vanedb_cpp.DiskStore(sys.argv[1])
        vector = store.get(42)
        operation = sys.argv[2]
        expected_error = TypeError if operation == "memoryview" else ValueError
        try:
            if operation == "direct":
                vector[0] = 99.0
            elif operation == "slice":
                vector[::2] = 99.0
            elif operation == "memoryview":
                memoryview(vector)[0] = 99.0
            elif operation == "ufunc_out":
                np.add(vector, np.float32(1), out=vector)
            elif operation == "copyto":
                np.copyto(vector, np.full_like(vector, 99.0))
            elif operation == "fill":
                vector.fill(99.0)
            else:
                raise AssertionError("unknown write operation")
        except expected_error as error:
            assert "read-only" in str(error).lower(), str(error)
            print("SAFE:", type(error).__name__, str(error), flush=True)
        else:
            raise AssertionError("write to read-only mmap unexpectedly succeeded")
        np.testing.assert_array_equal(vector, [1.0, 2.0, 3.0, 4.0])
    """)
    child = subprocess.run(
        [sys.executable, "-I", "-c", script, str(path), operation],
        capture_output=True, text=True, timeout=30,
    )
    assert child.returncode == 0, (
        f"{operation}: child returned {child.returncode}\n"
        f"stdout: {child.stdout}\nstderr: {child.stderr}"
    )
    assert "SAFE:" in child.stdout
    assert path.read_bytes() == before


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
