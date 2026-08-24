"""MEIA Python 包、资源与批处理入口的断代契约。"""

from __future__ import annotations

from importlib import import_module
from importlib.machinery import PathFinder
from importlib.resources import files
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _retired_package_name() -> str:
    return "".join(("at", "om", "v", "iz"))


def _retired_brand_pattern() -> re.Pattern[bytes]:
    first = b"at" + b"om"
    second = b"v" + b"iz"
    return re.compile(first + rb"[-_ ]?" + second, re.IGNORECASE)


def _retired_full_name_case() -> bytes:
    return b"Extended-" + b"Sys" + b"tem"


def test_meia_is_the_only_project_package_namespace():
    package = import_module("meia")

    assert package.__version__ == "0.11.0"
    assert (PROJECT_ROOT / "meia" / "__init__.py").is_file()
    assert not (PROJECT_ROOT / _retired_package_name()).exists()
    assert PathFinder.find_spec(
        _retired_package_name(),
        [str(PROJECT_ROOT)],
    ) is None


def test_meia_resources_load_from_the_new_package():
    from meia.brand import PRODUCT_FULL_NAME
    from meia.i18n import I18n, Locale

    assert PRODUCT_FULL_NAME == (
        "Molecular and Extended-system Illustration Assistant"
    )
    assert files("meia.locales").joinpath("en.json").is_file()
    assert I18n(Locale.EN).text("app.title.full") == PRODUCT_FULL_NAME


@pytest.mark.release
def test_meia_batch_module_is_the_only_documented_cli_entrypoint():
    result = subprocess.run(
        [sys.executable, "-m", "meia.batch", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout.lower()


def test_tracked_project_tree_uses_only_current_meia_naming():
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=True,
    ).stdout
    paths = [Path(os.fsdecode(item)) for item in listed.split(b"\0") if item]
    pattern = _retired_brand_pattern()
    retired_case = _retired_full_name_case()
    violations: list[tuple[str, str]] = []

    for relative_path in paths:
        encoded_path = os.fsencode(str(relative_path))
        if pattern.search(encoded_path):
            violations.append((str(relative_path), "path"))
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            continue
        content = path.read_bytes()
        if pattern.search(content):
            violations.append((str(relative_path), "content"))
        if retired_case in content:
            violations.append((str(relative_path), "full-name-case"))

    assert violations == []
