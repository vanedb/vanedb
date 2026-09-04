"""Check the description users receive in the installed wheel, not the checkout."""

from importlib.metadata import metadata
import re


def test_installed_package_has_markdown_description():
    package = metadata("vanedb")
    content_type = package.get("Description-Content-Type", "")
    assert content_type.partition(";")[0].strip().lower() == "text/markdown"
    assert package.get("Description", "").strip()


def test_installed_readme_python_examples():
    description = metadata("vanedb").get("Description", "")
    examples = re.findall(r"```python\n(.*?)\n```", description, re.DOTALL)
    assert examples, "The installed description must contain a runnable quick start"
    for example in examples:
        exec(compile(example, "<installed vanedb README>", "exec"), {})
