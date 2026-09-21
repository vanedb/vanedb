"""Type stubs for vanedb.

The runtime surface is a compiled PyO3 extension, so these stubs are what an
editor and a type checker see. `tests/test_type_stubs.py` asserts they match
the runtime module, because a stub that drifts is worse than no stub.
"""

import enum
import os
from collections.abc import Callable
from typing import Any, TypeAlias

__version__: str
__all__: list[str]

# Shape and dtype are validated at runtime, so these aliases document the
# accepted forms rather than constraining them — a checker cannot verify that
# a numpy array is 2-D float32.

#: A 1-D float32 buffer (numpy array, array.array, memoryview) or any sequence
#: of floats, with length equal to the index dimension.
VectorLike: TypeAlias = Any

#: A 2-D float32 buffer of shape (n, dim), or a sequence of float sequences.
BatchLike: TypeAlias = Any

#: A 1-D uint64 or int64 buffer, or a sequence of ints. Negative ids raise
#: ValueError.
IdsLike: TypeAlias = Any

#: A callable predicate taking an id (int) and returning bool.
FilterCallable: TypeAlias = Callable[[int], bool]

#: A filesystem path: `str` or any `os.PathLike`, `pathlib.Path` included.
PathLike: TypeAlias = str | os.PathLike[str]

class Metric(enum.IntEnum):
    """Distance metric for vector comparison.

    An IntEnum: `.name`, `.value`, `list(Metric)`, `Metric(1)`, hashing and
    pickling all work. The values are the on-disk metric field.

    Where an index constructor takes a Metric it also accepts any integer
    with `__index__` holding one of the values, NumPy integers and `bool`
    included (`True` is `Metric.COSINE`, as `Metric(True)` is); a float or a
    string is a TypeError and an integer naming no member a ValueError.
    """

    L2 = 0
    COSINE = 1
    DOT = 2

class FlatIndex:
    """Exact k-NN by brute-force scan. Thread-safe."""

    def __init__(self, dim: int, metric: Metric = ...) -> None: ...
    @property
    def metric(self) -> Metric: ...
    @property
    def dimension(self) -> int: ...
    def size(self) -> int:
        """An alias of `len(index)`, kept for C++ and JavaScript habits."""

    def __len__(self) -> int: ...
    def add(self, id: int, vector: VectorLike) -> None: ...
    def add_batch(self, ids: IdsLike, vectors: BatchLike) -> None: ...
    def get(self, id: int) -> list[float] | None:
        """The vector under `id`, or None if none is stored there."""

    def get_vector(self, id: int) -> list[float] | None:
        """The same read as `get`, under the other spelling."""

    def contains(self, id: int) -> bool: ...
    def remove(self, id: int) -> None:
        """Raises ValueError if no vector is stored under `id`."""

    def search(
        self,
        query: VectorLike,
        k: int,
        *,
        filter: FilterCallable | None = ...,
        allow_ids: IdsLike | None = ...,
        deny_ids: IdsLike | None = ...,
    ) -> list[tuple[int, float]]: ...

class ApproxIndex:
    """Approximate k-NN over an HNSW graph, with persistence."""

    def __init__(
        self,
        dim: int,
        metric: Metric = ...,
        capacity: int = ...,
        m: int = ...,
        ef_construction: int = ...,
        seed: int = ...,
    ) -> None: ...
    @property
    def metric(self) -> Metric: ...
    @property
    def dimension(self) -> int: ...
    @property
    def capacity(self) -> int: ...
    @property
    def m(self) -> int: ...
    @property
    def ef_construction(self) -> int: ...
    @property
    def seed(self) -> int: ...
    @property
    def ef_search(self) -> int: ...
    @ef_search.setter
    def ef_search(self, ef: int) -> None: ...
    def size(self) -> int:
        """An alias of `len(index)`, kept for C++ and JavaScript habits."""

    def __len__(self) -> int: ...
    def add(self, id: int, vector: VectorLike) -> None: ...
    def add_batch(self, ids: IdsLike, vectors: BatchLike) -> None: ...
    def upsert(self, id: int, vector: VectorLike) -> None: ...
    def get_vector(self, id: int) -> list[float] | None:
        """The vector under `id`, or None if none is stored there."""

    def get(self, id: int) -> list[float] | None:
        """The same read as `get_vector`, under the other spelling."""

    def contains(self, id: int) -> bool: ...
    def remove(self, id: int) -> None:
        """Raises ValueError if no vector is stored under `id`."""

    @property
    def tombstones(self) -> int: ...
    def compact(self) -> None: ...
    def search(
        self,
        query: VectorLike,
        k: int,
        *,
        ef_search: int | None = ...,
        max_ef_search: int | None = ...,
        filter: FilterCallable | None = ...,
        allow_ids: IdsLike | None = ...,
        deny_ids: IdsLike | None = ...,
    ) -> list[tuple[int, float]]: ...
    def save(self, path: PathLike) -> None: ...
    @staticmethod
    def load(path: PathLike) -> ApproxIndex:
        """Raises FileNotFoundError if the file is absent, ValueError if corrupt."""

    def to_bytes(self) -> bytes: ...
    @staticmethod
    def from_bytes(data: bytes) -> ApproxIndex:
        """Raises ValueError if the bytes are not a valid graph."""

class DiskIndexBuilder:
    """Builds a file that DiskIndex can memory-map."""

    def __init__(self, dim: int, metric: Metric = ...) -> None: ...
    @property
    def dimension(self) -> int: ...
    def size(self) -> int:
        """An alias of `len(builder)`, kept for C++ and JavaScript habits."""

    def __len__(self) -> int: ...
    def add(self, id: int, vector: VectorLike) -> None: ...
    def save(self, path: PathLike) -> None: ...

class DiskIndex:
    """Memory-mapped exact k-NN. Open with DiskIndex.open; not constructible.

    Keep the underlying file's contents and length unchanged in every process,
    from before open until the index is released. In-place modification can
    corrupt results or crash the interpreter. DiskIndexBuilder.save atomically
    replaces the path while leaving existing mappings intact.
    """

    @property
    def metric(self) -> Metric: ...
    @property
    def dimension(self) -> int: ...
    def size(self) -> int:
        """An alias of `len(index)`, kept for C++ and JavaScript habits."""

    def __len__(self) -> int: ...
    def get(self, id: int) -> list[float] | None:
        """The vector under `id`, or None if none is stored there."""

    def get_vector(self, id: int) -> list[float] | None:
        """The same read as `get`, under the other spelling."""

    def contains(self, id: int) -> bool: ...
    def search(
        self,
        query: VectorLike,
        k: int,
        *,
        filter: FilterCallable | None = ...,
        allow_ids: IdsLike | None = ...,
        deny_ids: IdsLike | None = ...,
    ) -> list[tuple[int, float]]: ...
    @staticmethod
    def open(path: PathLike) -> DiskIndex:
        """Raises FileNotFoundError if the file is absent, ValueError if corrupt."""
