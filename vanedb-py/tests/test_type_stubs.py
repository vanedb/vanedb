"""The stub must match the runtime module.

A stub that disagrees with the runtime is worse than no stub: the editor
confidently autocompletes a method that does not exist.
"""

import ast
import inspect
import pathlib

import vanedb

# Resolved through the imported package, not the repo layout: CI copies this
# suite outside the tree to test the built wheel, and the stub that ships is
# the one worth checking anyway.
STUB = pathlib.Path(vanedb.__file__).resolve().parent / "__init__.pyi"


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


def _stub_properties() -> dict[str, set[str]]:
    """Class name -> the members the stub decorates with @property."""
    classes = {}
    for node in _stub_tree().body:
        if isinstance(node, ast.ClassDef):
            props = {
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef)
                and any(
                    isinstance(d, ast.Name) and d.id == "property"
                    for d in item.decorator_list
                )
            }
            classes[node.name] = props
    return classes


def _runtime_members(obj: type) -> set[str]:
    return {
        name
        for name in dir(obj)
        if not name.startswith("_") or name in {"__len__", "__init__"}
    }


def test_the_stub_and_marker_ship_in_the_package():
    assert STUB.is_file(), f"no __init__.pyi in {STUB.parent}"
    assert (STUB.parent / "py.typed").is_file(), f"no py.typed in {STUB.parent}"


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


def test_the_stub_marks_a_property_exactly_where_the_runtime_has_one():
    """Names alone are not enough: `x.dimension` and `x.dimension()` are
    different programs, and only one of them runs.

    A `#[getter]` reaches Python as a getset descriptor; a plain method
    reaches it as a method descriptor. Comparing only member names let an
    edit strip `@property` off three `dimension` declarations while every
    other stub test stayed green, so the stub told editors to call a value.
    """
    for cls_name, stub_props in _stub_properties().items():
        runtime = getattr(vanedb, cls_name, None)
        if runtime is None:
            continue
        runtime_props = {
            name
            for name in _runtime_members(runtime)
            if inspect.isdatadescriptor(getattr(runtime, name, None))
        }
        assert stub_props == runtime_props, (
            f"{cls_name}: stub says property {sorted(stub_props)}, "
            f"runtime has {sorted(runtime_props)}"
        )


def test_no_stub_member_is_declared_twice():
    """`@property` twice on one member is `property(property(f))`, which is
    not a property. The duplicate that motivated this shipped in a release
    candidate because nothing counted declarations.
    """
    for node in _stub_tree().body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                names = [
                    d.id for d in item.decorator_list if isinstance(d, ast.Name)
                ]
                assert len(names) == len(set(names)), (
                    f"{node.name}.{item.name} repeats a decorator: {names}"
                )
