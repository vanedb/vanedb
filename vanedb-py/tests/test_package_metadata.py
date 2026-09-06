"""Check what users receive in the installed wheel, not what is in the checkout."""

from importlib.metadata import files, metadata
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


def test_installed_wheel_ships_the_license_text():
    """MIT requires the notice in copies; a link in the description is not one."""
    package = metadata("vanedb")
    assert package.get("License-Expression") == "MIT"

    licenses = [f for f in files("vanedb") or [] if "licenses/LICENSE" in str(f)]
    assert licenses, "the wheel must contain the LICENSE text, not just a link"
    assert "MIT License" in licenses[0].read_text()
