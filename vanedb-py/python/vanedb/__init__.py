"""Embeddable vector database for edge AI."""

# The compiled extension is the whole implementation; this module exists so the
# package can also ship py.typed and __init__.pyi, which a maturin-generated
# __init__ cannot carry. __all__ and __version__ are re-exported explicitly
# because a star import only brings what __all__ names.
from .vanedb import *  # noqa: F403
from .vanedb import __all__, __version__  # noqa: F401
