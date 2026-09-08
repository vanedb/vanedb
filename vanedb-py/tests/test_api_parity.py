"""Parity between the Python surface and the Rust core it wraps.

Each gap here was found by a cross-platform audit, and each is the same
shape: the core did the work and documented why, and the binding dropped
it. Nothing in the suite asserted that a design decision recorded in the
core survives the boundary, so nothing noticed.
"""

import inspect
import threading

import pytest

from vanedb import ApproxIndex, DiskIndex, DiskIndexBuilder, FlatIndex, Metric


def test_both_spellings_of_the_read_exist_on_every_index(tmp_path):
    """`get` and `get_vector` are one operation under two names so a program
    is not tied to one index type (#85). Python had `get` on FlatIndex and
    `get_vector` on ApproxIndex and neither on the other, so outgrowing
    FlatIndex meant an AttributeError at every call site."""
    vector = [3.0, 4.0]

    flat = FlatIndex(2, Metric.L2)
    flat.add(1, vector)
    assert flat.get(1) == vector
    assert flat.get_vector(1) == vector

    approx = ApproxIndex(2, Metric.L2)
    approx.add(1, vector)
    assert approx.get(1) == vector
    assert approx.get_vector(1) == vector

    builder = DiskIndexBuilder(2, Metric.L2)
    builder.add(1, vector)
    path = str(tmp_path / "store.vndb")
    builder.save(path)
    disk = DiskIndex.open(path)
    assert disk.get(1) == vector
    assert disk.get_vector(1) == vector


def test_search_takes_a_per_query_beam_width():
    """Raising recall for one query used to mean assigning to the shared
    `ef_search` property. Every search releases the GIL, so a concurrent
    thread could observe the mutation -- the exact bug the C ABI was changed
    to fix, and what SearchParams exists to prevent."""
    index = ApproxIndex(1, Metric.L2, capacity=64, m=4, ef_construction=32, seed=7)
    for i in range(40):
        index.add(i, [float(i)])
    index.ef_search = 4

    wide = index.search([0.0], 10, ef_search=40)
    assert [r[0] for r in wide] == list(range(10))
    assert index.ef_search == 4, "a per-query beam must not mutate the index"

    # Omitting it uses the stored value, exactly as before.
    assert len(index.search([0.0], 10)) == 10
    assert index.ef_search == 4


def test_a_per_query_beam_is_not_visible_to_another_thread():
    """The property is shared state; the argument is not."""
    index = ApproxIndex(1, Metric.L2, capacity=256, m=4, ef_construction=32, seed=7)
    for i in range(200):
        index.add(i, [float(i)])
    index.ef_search = 10

    observed = []
    stop = threading.Event()

    def watcher():
        while not stop.is_set():
            observed.append(index.ef_search)

    t = threading.Thread(target=watcher)
    t.start()
    try:
        for _ in range(200):
            index.search([0.0], 5, ef_search=200)
    finally:
        stop.set()
        t.join()

    assert set(observed) == {10}, f"another thread saw {sorted(set(observed))}"


def test_a_loaded_index_reports_the_geometry_it_was_built_with(tmp_path):
    """The core exposes m/ef_construction/seed because "a caller that did not
    build it has no other way to know what graph they are searching". That
    argument is about `load`, which Python has."""
    index = ApproxIndex(3, Metric.COSINE, capacity=512, m=6, ef_construction=48, seed=1234)
    assert index.m == 6
    assert index.ef_construction == 48
    assert index.seed == 1234

    index.add(1, [1.0, 0.0, 0.0])
    path = str(tmp_path / "graph.vndb")
    index.save(path)

    loaded = ApproxIndex.load(path)
    assert loaded.m == 6
    assert loaded.ef_construction == 48
    assert loaded.seed == 1234
    assert loaded.capacity == 512
    assert loaded.metric == Metric.COSINE


@pytest.mark.parametrize("name", ["m", "ef_construction", "seed"])
def test_the_geometry_accessors_are_properties_not_methods(name):
    """A read-only scalar is a property everywhere else on this class; a bound
    method here would be a silently truthy object in a comparison. `tombstones`
    shipped as a method once for exactly this reason, so the check is the one
    test_type_stubs.py settled on: a PyO3 #[getter] is a getset_descriptor,
    which is a data descriptor, while a method_descriptor is not."""
    assert inspect.isdatadescriptor(getattr(ApproxIndex, name))


def test_a_missing_id_raises_keyerror_on_every_index(tmp_path):
    """A lookup miss is KeyError in Python. It used to be ValueError, which
    could not be separated from a dimension mismatch without parsing English.

    KeyError subclasses LookupError, not ValueError, so this is a breaking
    change -- made before the first publish rather than never."""
    flat = FlatIndex(2, Metric.L2)
    flat.add(1, [1.0, 2.0])
    with pytest.raises(KeyError):
        flat.get(99)
    with pytest.raises(KeyError):
        flat.remove(99)

    approx = ApproxIndex(2, Metric.L2)
    approx.add(1, [1.0, 2.0])
    with pytest.raises(KeyError):
        approx.get_vector(99)
    with pytest.raises(KeyError):
        approx.remove(99)

    builder = DiskIndexBuilder(2, Metric.L2)
    builder.add(1, [1.0, 2.0])
    path = str(tmp_path / "store.vndb")
    builder.save(path)
    with pytest.raises(KeyError):
        DiskIndex.open(path).get(99)


def test_validation_failures_are_still_valueerror():
    """Only the lookup miss moved. A wrong dimension is still a ValueError,
    so the two are now distinguishable by type."""
    flat = FlatIndex(2, Metric.L2)
    flat.add(1, [1.0, 2.0])

    with pytest.raises(ValueError):
        flat.add(2, [1.0])
    with pytest.raises(ValueError):
        flat.add(1, [1.0, 2.0])          # duplicate id
    with pytest.raises(ValueError):
        flat.search([1.0, 2.0], 0)
    assert not isinstance(ValueError(), KeyError)


def test_keyerror_message_is_readable():
    """KeyError's repr quotes its argument, so the message must still name the
    id rather than arriving as a bare number."""
    flat = FlatIndex(2, Metric.L2)
    with pytest.raises(KeyError) as excinfo:
        flat.get(42)
    assert "42" in str(excinfo.value)
