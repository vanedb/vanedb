"""The Python error contract.

A missing index file, a corrupt one and a bad argument must be
distinguishable without reading the message, and every out-of-range integer must
raise ValueError -- OverflowError is not a ValueError subclass, so
`except ValueError` would miss it.
"""

import sys

import pytest

import vanedb


def test_missing_index_file_raises_filenotfounderror(tmp_path):
    missing = tmp_path / "does-not-exist.vane"
    with pytest.raises(FileNotFoundError):
        vanedb.ApproxIndex.load(str(missing))


def test_load_or_build_is_expressible_with_ordinary_python(tmp_path):
    # The motivating pattern, written the way a user would write it.
    path = str(tmp_path / "index.vane")
    try:
        index = vanedb.ApproxIndex.load(path)
    except FileNotFoundError:
        index = vanedb.ApproxIndex(dim=4)
    index.add(1, [1.0, 0.0, 0.0, 0.0])
    assert len(index) == 1


def test_corrupt_index_file_raises_valueerror_not_filenotfound(tmp_path):
    path = tmp_path / "garbage.vane"
    path.write_bytes(b"this is not a vanedb file")
    with pytest.raises(ValueError) as excinfo:
        vanedb.ApproxIndex.load(str(path))
    assert not isinstance(excinfo.value, FileNotFoundError)


def test_a_dimension_mismatch_is_still_a_valueerror():
    index = vanedb.ApproxIndex(dim=4)
    with pytest.raises(ValueError):
        index.add(1, [1.0, 2.0])


@pytest.mark.parametrize(
    "call",
    [
        lambda idx: idx.add(-1, [1.0, 0.0, 0.0, 0.0]),
        lambda idx: idx.get_vector(-1),
        lambda idx: idx.remove(-1),
        lambda idx: idx.upsert(-1, [1.0, 0.0, 0.0, 0.0]),
    ],
    ids=["add", "get_vector", "remove", "upsert"],
)
def test_a_negative_id_is_a_valueerror_everywhere(call):
    # OverflowError is not a ValueError subclass, so raising it here would
    # slip past `except ValueError`.
    index = vanedb.ApproxIndex(dim=4)
    with pytest.raises(ValueError):
        call(index)


def test_a_negative_id_in_a_plain_list_batch_is_a_valueerror():
    # A plain list takes a different conversion path from a numpy array and
    # must report identically.
    index = vanedb.ApproxIndex(dim=2)
    with pytest.raises(ValueError):
        index.add_batch([1, -1], [[1.0, 0.0], [0.0, 1.0]])


def test_a_negative_id_in_a_numpy_batch_is_a_valueerror():
    numpy = pytest.importorskip("numpy")
    index = vanedb.ApproxIndex(dim=2)
    ids = numpy.array([1, -1], dtype=numpy.int64)
    vectors = numpy.array([[1.0, 0.0], [0.0, 1.0]], dtype=numpy.float32)
    with pytest.raises(ValueError):
        index.add_batch(ids, vectors)


def test_contains_rejects_a_negative_id_consistently():
    index = vanedb.ApproxIndex(dim=4)
    with pytest.raises(ValueError):
        index.contains(-1)


def test_an_id_above_u64_max_is_a_valueerror_not_overflowerror():
    # Out of range in the other direction: too large for i64 as well, which
    # must still be a ValueError rather than an OverflowError.
    index = vanedb.ApproxIndex(dim=4)
    for call in (
        lambda: index.add(2**64, [1.0, 0.0, 0.0, 0.0]),
        lambda: index.get_vector(2**64),
        lambda: index.contains(2**70),
    ):
        with pytest.raises(ValueError):
            call()


def test_an_out_of_range_id_in_a_list_batch_is_a_valueerror():
    index = vanedb.ApproxIndex(dim=2)
    with pytest.raises(ValueError):
        index.add_batch([1, 2**64], [[1.0, 0.0], [0.0, 1.0]])


INVALID_SIZES = [
    pytest.param(-1, ValueError, id="negative"),
    pytest.param(-(2**100), ValueError, id="huge_negative"),
    pytest.param(2 * (sys.maxsize + 1), ValueError, id="above_usize"),
    pytest.param(2**100, ValueError, id="huge_positive"),
    pytest.param(1.5, TypeError, id="float"),
    pytest.param("1", TypeError, id="string"),
    pytest.param(None, TypeError, id="none"),
]


@pytest.mark.parametrize("value,error", INVALID_SIZES)
@pytest.mark.parametrize(
    "construct",
    [
        lambda value: vanedb.FlatIndex(value, vanedb.Metric.L2),
        lambda value: vanedb.ApproxIndex(value, vanedb.Metric.L2),
        lambda value: vanedb.ApproxIndex(3, vanedb.Metric.L2, capacity=value),
        lambda value: vanedb.ApproxIndex(3, vanedb.Metric.L2, m=value),
        lambda value: vanedb.ApproxIndex(3, vanedb.Metric.L2, ef_construction=value),
        lambda value: vanedb.DiskIndexBuilder(value, vanedb.Metric.L2),
    ],
    ids=["flat_dim", "approx_dim", "capacity", "m", "ef_construction", "disk_dim"],
)
def test_invalid_constructor_sizes(construct, value, error):
    with pytest.raises(error):
        construct(value)


@pytest.mark.parametrize("value,error", INVALID_SIZES)
@pytest.mark.parametrize("kind", ["flat", "approx", "disk"])
def test_invalid_search_k(tmp_path, kind, value, error):
    if kind == "disk":
        builder = vanedb.DiskIndexBuilder(2, vanedb.Metric.L2)
        builder.add(1, [1.0, 0.0])
        path = str(tmp_path / "index.vane")
        builder.save(path)
        index = vanedb.DiskIndex.open(path)
    else:
        cls = vanedb.FlatIndex if kind == "flat" else vanedb.ApproxIndex
        index = cls(2, vanedb.Metric.L2)
        index.add(1, [1.0, 0.0])
    with pytest.raises(error):
        index.search([1.0, 0.0], value)
    assert index.search([1.0, 0.0], 1) == [(1, 0.0)]
    with pytest.raises(ValueError):
        index.search([1.0, 0.0], 0)


@pytest.mark.parametrize("value,error", INVALID_SIZES)
def test_invalid_ef_search_preserves_setting(value, error):
    index = vanedb.ApproxIndex(2, vanedb.Metric.L2, capacity=1)
    index.add(1, [1.0, 0.0])
    index.ef_search = 73
    with pytest.raises(error):
        index.ef_search = value
    assert index.ef_search == 73
    assert index.search([1.0, 0.0], 1) == [(1, 0.0)]
    index.ef_search = 0
    assert index.ef_search == 0


@pytest.mark.parametrize(
    "value,error",
    [
        (-1, ValueError),
        (-(2**100), ValueError),
        (2**64, ValueError),
        (2**100, ValueError),
        (1.5, TypeError),
        ("1", TypeError),
        (None, TypeError),
    ],
)
def test_invalid_seed(value, error):
    with pytest.raises(error):
        vanedb.ApproxIndex(2, seed=value, capacity=1)


@pytest.mark.parametrize("seed", [0, 2**64 - 1])
def test_seed_accepts_unsigned_boundaries(seed):
    index = vanedb.ApproxIndex(2, seed=seed, capacity=1)
    index.add(1, [1.0, 0.0])
    assert index.search([1.0, 0.0], 1) == [(1, 0.0)]


def test_an_absurd_dimension_is_a_valueerror_not_a_panic():
    """`rows * dim` overflowed into a capacity panic, escaping the error model.

    `FlatIndex::new` accepted a dimension `ApproxIndexBuilder::build` rejects,
    so the failure surfaced later as a PanicException rather than a ValueError.
    """
    with pytest.raises(ValueError):
        vanedb.FlatIndex(2**62, vanedb.Metric.L2)
    with pytest.raises(ValueError):
        vanedb.ApproxIndex(2**62, vanedb.Metric.L2)
    with pytest.raises(ValueError):
        vanedb.DiskIndexBuilder(2**62, vanedb.Metric.L2)
