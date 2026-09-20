"""Embeddable vector database for edge AI."""

# The compiled extension is most of the implementation; this module exists so
# the package can also ship py.typed and __init__.pyi, which a maturin-generated
# __init__ cannot carry, and so `Metric` can be a real `enum.IntEnum`. __all__
# and __version__ are re-exported explicitly because a star import only brings
# what __all__ names.
import enum

from .vanedb import *  # noqa: F403
from .vanedb import __all__ as _core_all
from .vanedb import __version__  # noqa: F401


class Metric(enum.IntEnum):
    """Distance metric for vector comparison.

    ``L2`` is squared Euclidean distance, ``COSINE`` cosine distance and
    ``DOT`` negative dot product; lower distances rank first. The values are
    the on-disk ``metric`` field and the C ABI's ``uint32_t``, so
    ``int(Metric.COSINE)`` is what a VNDB header stores.

    An ``IntEnum`` like any other Python enum (RFC 0011): ``.name``,
    ``.value``, ``list(Metric)``, ``Metric(1)``, hashing and pickling all
    work, and ``Metric.L2 == 0`` is true.
    """

    L2 = 0
    COSINE = 1
    DOT = 2


# `Metric` lives here rather than in the extension, so it is added to the
# extension's list rather than star-imported from it. The index constructors
# and `metric` properties look this class up by name.
__all__ = ["Metric", *_core_all]
