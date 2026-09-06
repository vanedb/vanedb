"""The Python error contract.

Every failure used to surface as ValueError, so "the index file isn't there
yet" was indistinguishable from "the vector had the wrong length" without
reading the message. And negative ids escaped as OverflowError, which is not
a ValueError subclass, so `except ValueError` silently missed them.
"""

import pathlib

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
    # OverflowError is not a ValueError subclass, so `except ValueError`
    # used to miss these entirely.
    index = vanedb.ApproxIndex(dim=4)
    with pytest.raises(ValueError):
        call(index)


def test_a_negative_id_in_a_plain_list_batch_is_a_valueerror():
    # Only the numpy path was checked; a plain list went through a different
    # conversion and raised OverflowError.
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
    # The negative case was fixed by retrying the conversion as i64, but a
    # value too large for i64 as well still escaped as OverflowError, which
    # `except ValueError` does not catch.
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
