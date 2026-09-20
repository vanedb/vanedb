"""Parity between the Python surface and the Rust core it wraps.

Each gap here was found by a cross-platform audit, and each is the same
shape: the core did the work and documented why, and the binding dropped
it. Nothing in the suite asserted that a design decision recorded in the
core survives the boundary, so nothing noticed.
"""

import inspect
import random
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


def _clustered_index(seed=11):
    """A graph where the beam width actually decides the answer.

    The first version of this test used 40 one-dimensional points and could
    not fail: `search` computes `max(ef_search, k)`, so a stored `ef_search`
    of 4 was raised to `k = 10`, and a 40-node 1-D graph is exhaustively
    correct at any beam. It passed whether the keyword did anything or not.

    2000 vectors in 32 dimensions, with `m` and `ef_construction` low enough
    that the graph is genuinely sparse, separates the two.
    """
    rng = random.Random(seed)
    rows = [(i, [rng.gauss(0, 1) for _ in range(32)]) for i in range(2000)]
    index = ApproxIndex(32, Metric.L2, capacity=2048, m=4, ef_construction=8, seed=seed)
    for vid, vec in rows:
        index.add(vid, vec)
    return index, rows


def _brute_force(rows, query, k):
    """Ground truth, computed here rather than asked of the index."""
    scored = [(sum((a - b) ** 2 for a, b in zip(v, query)), vid) for vid, v in rows]
    scored.sort()
    return {vid for _, vid in scored[:k]}


def test_search_takes_a_per_query_beam_width():
    """Raising recall for one query used to mean assigning to the shared
    `ef_search` property. Every search releases the GIL, so a concurrent
    thread could observe the mutation -- the exact bug the C ABI was changed
    to fix, and what SearchParams exists to prevent."""
    index, rows = _clustered_index()
    index.ef_search = 10
    query = [0.0] * 32
    truth = _brute_force(rows, query, 10)

    narrow = {r[0] for r in index.search(query, 10)}
    wide = {r[0] for r in index.search(query, 10, ef_search=400)}

    narrow_recall = len(narrow & truth) / 10
    wide_recall = len(wide & truth) / 10
    assert wide_recall > narrow_recall + 0.3, (
        f"a wider beam must find more true neighbours: "
        f"stored ef=10 gave {narrow_recall}, ef_search=400 gave {wide_recall}. "
        f"If these are equal the keyword is being ignored."
    )
    assert index.ef_search == 10, "a per-query beam must not mutate the index"


def test_a_per_query_beam_is_not_visible_to_another_thread():
    """The property is shared state; the argument is not."""
    index = ApproxIndex(1, Metric.L2, capacity=256, m=4, ef_construction=32, seed=7)
    for i in range(200):
        index.add(i, [float(i)])
    index.ef_search = 10

    # A set, not a list: appending every sample allocated ~290 MB on a fast
    # runner and scaled with the runner's speed. The assertion only needs the
    # distinct values ever seen.
    observed = set()
    stop = threading.Event()

    def watcher():
        while not stop.is_set():
            observed.add(index.ef_search)

    t = threading.Thread(target=watcher)
    t.start()
    try:
        for _ in range(200):
            index.search([0.0], 5, ef_search=200)
    finally:
        stop.set()
        t.join()

    assert observed == {10}, f"another thread saw {sorted(observed)}"


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


def test_a_lookup_miss_is_none_and_a_removal_miss_is_valueerror(tmp_path):
    """A lookup miss is a value, not an error (RFC 0011): `get` and
    `get_vector` return `None` on every index type, the way `dict.get` and
    every keyed store in the roadmap's precedent survey do. `remove` of a
    missing id stays an error, because a caller that removes what is not
    there has a bug and `remove` is not named `get` -- and with `KeyError`
    gone from the binding it is a `ValueError`, the validation bucket.

    All eight reads and both removes, including the alias spellings.
    """
    flat = FlatIndex(2, Metric.L2)
    flat.add(1, [1.0, 2.0])

    approx = ApproxIndex(2, Metric.L2)
    approx.add(1, [1.0, 2.0])

    builder = DiskIndexBuilder(2, Metric.L2)
    builder.add(1, [1.0, 2.0])
    path = str(tmp_path / "store.vndb")
    builder.save(path)
    disk = DiskIndex.open(path)

    for index in (flat, approx, disk):
        name = type(index).__name__
        assert index.get(1) == [1.0, 2.0], name
        assert index.get_vector(1) == [1.0, 2.0], name
        assert index.get(99) is None, name
        assert index.get_vector(99) is None, name
        assert not index.contains(99), name

    for index in (flat, approx):
        with pytest.raises(ValueError, match="99") as excinfo:
            index.remove(99)
        assert not isinstance(excinfo.value, KeyError), type(index).__name__


def test_validation_failures_are_still_valueerror():
    """A wrong dimension, a duplicate id and a zero `k` are ValueError, as
    they have always been; a lookup miss is no exception at all."""
    flat = FlatIndex(2, Metric.L2)
    flat.add(1, [1.0, 2.0])

    with pytest.raises(ValueError):
        flat.add(2, [1.0])
    with pytest.raises(ValueError):
        flat.add(1, [1.0, 2.0])          # duplicate id
    with pytest.raises(ValueError):
        flat.search([1.0, 2.0], 0)


def test_the_vocabulary_of_rfc_0011(tmp_path):
    """The Python column of `conformance/vocabulary/README.md`, in one place:
    a lookup miss is `None` under both read spellings; the count is
    `len(index)` with `size()` kept as an alias; emptiness is truth testing;
    the beam default is the `ef_search` property; `contains` stays; `remove`
    of a missing id stays an error."""
    flat = FlatIndex(2, Metric.L2)
    approx = ApproxIndex(2, Metric.L2, capacity=4)
    builder = DiskIndexBuilder(2, Metric.L2)
    for empty in (flat, approx, builder):
        assert len(empty) == 0 and empty.size() == 0
        assert not empty, type(empty).__name__
    for index in (flat, approx, builder):
        index.add(7, [1.0, 0.0])
        assert len(index) == index.size() == 1
        assert index, type(index).__name__
    path = str(tmp_path / "vocabulary.vndb")
    builder.save(path)
    disk = DiskIndex.open(path)
    assert len(disk) == disk.size() == 1 and disk
    for index in (flat, approx, disk):
        assert index.get(7) == index.get_vector(7) == [1.0, 0.0]
        assert index.get(8) is index.get_vector(8) is None
        assert index.contains(7) and not index.contains(8)
    with pytest.raises(ValueError):
        approx.remove(8)
    assert approx.ef_search == 50, "documented default"
    approx.ef_search = 64
    assert approx.ef_search == 64


def test_metric_is_an_int_enum():
    """`Metric` used to be a PyO3 class with `eq_int` and nothing else: no
    `.name`, no `.value`, not iterable, unlike every other enum a Python user
    has met. It is an `enum.IntEnum` now (RFC 0011), so all of that comes from
    the standard library, and the wire values still pin the on-disk field."""
    import enum

    assert issubclass(Metric, enum.IntEnum)
    assert [m.name for m in Metric] == ["L2", "COSINE", "DOT"]
    assert [m.value for m in Metric] == [0, 1, 2]
    assert Metric(1) is Metric.COSINE
    assert Metric["DOT"] is Metric.DOT
    assert Metric.L2 == 0 and isinstance(Metric.L2, int)
    assert {Metric.COSINE: "c"}[1] == "c", "hashes like its value"
    assert Metric.__module__ == "vanedb", "pickles by importable name"
    with pytest.raises(ValueError):
        Metric(7)

    # The constructors take a member or its integer value, the way an IntEnum
    # itself does, and report the member back.
    assert FlatIndex(2, Metric.COSINE).metric is Metric.COSINE
    assert FlatIndex(2, 1).metric is Metric.COSINE
    assert ApproxIndex(2).metric is Metric.L2
    with pytest.raises(ValueError):
        FlatIndex(2, 7)
    with pytest.raises(TypeError):
        FlatIndex(2, "cosine")


def test_distances_are_real_values_not_zero(tmp_path):
    """Replacing every returned distance with 0.0 passed all 164 tests.

    Every distance assertion in the suite was `< 1e-6`, `== 0.0`, or a
    comparison between two outputs of the same code path, so none of them
    could tell a working metric from a constant. These check magnitudes
    against arithmetic done here.
    """
    a, b = [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]

    flat = FlatIndex(3, Metric.L2)
    flat.add(1, a)
    flat.add(2, b)
    hits = dict(flat.search(a, 2))
    assert hits[1] == pytest.approx(0.0, abs=1e-6)
    assert hits[2] == pytest.approx(2.0, rel=1e-5), "L2 is squared: |a-b|^2 = 2"

    cos = FlatIndex(3, Metric.COSINE)
    cos.add(1, a)
    cos.add(2, b)
    hits = dict(cos.search(a, 2))
    assert hits[1] == pytest.approx(0.0, abs=1e-6)
    assert hits[2] == pytest.approx(1.0, rel=1e-5), "orthogonal cosine distance is 1"

    dot = FlatIndex(3, Metric.DOT)
    dot.add(1, [2.0, 0.0, 0.0])
    dot.add(2, b)
    hits = dict(dot.search(a, 2))
    assert hits[1] == pytest.approx(-2.0, rel=1e-5), "dot is negated: -(a.b)"
    assert hits[2] == pytest.approx(0.0, abs=1e-6)

    approx = ApproxIndex(3, Metric.L2)
    approx.add(1, a)
    approx.add(2, b)
    assert dict(approx.search(a, 2))[2] == pytest.approx(2.0, rel=1e-5)

    builder = DiskIndexBuilder(3, Metric.L2)
    builder.add(1, a)
    builder.add(2, b)
    path = str(tmp_path / "d.vndb")
    builder.save(path)
    assert dict(DiskIndex.open(path).search(a, 2))[2] == pytest.approx(2.0, rel=1e-5)


def test_metric_wire_values_are_pinned():
    """The suite only ever compared `Metric` to `Metric`, so swapping the
    integer discriminants (L2=2, Cosine=0, Dot=1) changed nothing. Those
    integers are the on-disk `metric` field and the C ABI's contract; the C
    side pins them and Python did not."""
    assert int(Metric.L2) == 0
    assert int(Metric.COSINE) == 1
    assert int(Metric.DOT) == 2


def test_a_saved_index_restores_its_search_beam(tmp_path):
    """`ApproxIndex.load` discarding the persisted `ef_search` survived the
    suite. It travels in the file so a tuned index ships as one artifact."""
    index = ApproxIndex(2, Metric.L2, capacity=32, m=4, ef_construction=16, seed=3)
    index.add(1, [1.0, 0.0])
    index.ef_search = 137
    path = str(tmp_path / "g.vndb")
    index.save(path)
    assert ApproxIndex.load(path).ef_search == 137
