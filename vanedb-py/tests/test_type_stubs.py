"""The stub must match the runtime module.

A stub that has drifted is worse than no stub: the editor confidently
autocompletes a method that no longer exists. The READMEs drifted this way
before a guard was added, so this one is checked in CI (#105).
"""

import ast
import pathlib

import vanedb

STUB = pathlib.Path(__file__).resolve().parents[1] / "python" / "vanedb" / "__init__.pyi"


def _stub_tree() -> ast.Module:
    return ast.parse(STUB.read_text(encoding="utf-8"))


def _stub_classes() -> dict[str, set[str]]:
    """Class name -> the members the stub declares for it."""
    classes = {}
    for node in _stub_tree().body:
        if isinstance(node, ast.ClassDef):
            members = set()
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    members.add(item.name)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    members.add(item.target.id)
            classes[node.name] = members
    return classes


def _runtime_members(obj: type) -> set[str]:
    return {
        name
        for name in dir(obj)
        if not name.startswith("_") or name in {"__len__", "__init__"}
    }


def test_the_stub_file_ships_in_the_package():
    assert STUB.is_file(), "no __init__.pyi"
    assert (STUB.parent / "py.typed").is_file(), "no py.typed marker"


def test_the_stub_declares_every_exported_name():
    stubbed = set(_stub_classes()) | {
        node.target.id
        for node in _stub_tree().body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    for name in vanedb.__all__:
        assert name in stubbed, f"{name} is exported but missing from the stub"


def test_no_stubbed_class_is_missing_a_runtime_member():
    for cls_name, stubbed in _stub_classes().items():
        runtime = getattr(vanedb, cls_name, None)
        if runtime is None:
            continue
        missing = _runtime_members(runtime) - stubbed
        # DiskIndex has no constructor; __init__ is inherited from object.
        missing.discard("__init__")
        assert not missing, f"{cls_name} gained {sorted(missing)}; the stub is stale"


def test_no_stubbed_member_has_disappeared_from_the_runtime():
    for cls_name, stubbed in _stub_classes().items():
        runtime = getattr(vanedb, cls_name, None)
        assert runtime is not None, f"stub declares {cls_name}, which does not exist"
        extra = {m for m in stubbed if not hasattr(runtime, m)}
        assert not extra, f"{cls_name} stub declares {sorted(extra)}, which the runtime lacks"


def test_the_package_is_marked_typed_at_runtime():
    package_dir = pathlib.Path(vanedb.__file__).parent
    assert (package_dir / "py.typed").is_file(), "py.typed missing from the built package"
