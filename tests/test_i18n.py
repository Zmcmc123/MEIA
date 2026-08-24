"""MEIA 双语词典与格式化契约。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from meia.i18n import I18n, Locale, LocalizedError, validate_catalog_pair


def test_catalogs_have_identical_keys_and_placeholders():
    assert validate_catalog_pair() is None


def test_i18n_formats_both_locales_without_cross_language_fallback():
    assert I18n(Locale.ZH_CN).text("structure.atom_count", count=1) == "1 个原子"
    assert I18n(Locale.EN).text("structure.atom_count", count=1) == "1 atom"
    assert I18n(Locale.EN).text("structure.atom_count", count=2) == "2 atoms"


def test_i18n_bundle_strips_namespace_but_preserves_templates():
    bundle = I18n(Locale.EN).bundle("viewer")
    assert bundle["camera.apply"] == "Apply Current View"
    assert bundle["selection.count.one"] == (
        "Temporary selection: {count} atom{pending}"
    )
    assert all(not key.startswith("viewer.") for key in bundle)


def test_localized_error_uses_stable_code_and_keeps_parameters():
    error = LocalizedError(
        "radius must be positive",
        message_key="atom.invalid_radius",
        message_params={"symbol": "O", "value": -0.1},
    )
    assert I18n(Locale.EN).error_text(error, "errors.operation_failed") == (
        "O display radius must be greater than 0 Å; received -0.1."
    )


def test_untyped_error_keeps_diagnostic_detail():
    assert I18n(Locale.EN).error_text(
        RuntimeError("boom"),
        "errors.operation_failed",
    ) == "Operation failed: RuntimeError: boom"


def test_unknown_error_boundary_preserves_original_diagnostic_and_user_path():
    rendered = I18n(Locale.EN).error_text(
        ValueError("无法读取 /tmp/中文构型.xyz：ASE parser failed"),
        "errors.operation_failed",
    )
    assert rendered == (
        "Operation failed: ValueError: 无法读取 /tmp/中文构型.xyz："
        "ASE parser failed"
    )


def test_missing_translation_key_fails_loudly():
    with pytest.raises(KeyError, match="missing.key"):
        I18n(Locale.EN).text("missing.key")


def test_radius_profile_help_matches_confirmed_bilingual_copy():
    assert I18n(Locale.ZH_CN).text("atom.radius_mode.help") == (
        "共价半径与相等半径分别保存一套原子大小和化学键粗细的默认参数；"
        "切换方案后，点击“应用原子设置”即可整套应用对应参数。"
    )
    assert I18n(Locale.EN).text("atom.radius_mode.help") == (
        "Covalent Radii and Uniform Radii each store a separate default set "
        "of atom-size and bond-thickness parameters. After switching modes, "
        "click “Apply Atom Settings” to apply the corresponding set."
    )


def test_literal_catalog_calls_exist_in_both_locales():
    project_root = Path(__file__).resolve().parents[1]
    keys = set()
    for path in (project_root / "app.py", project_root / "meia" / "sidebar.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr != "text" or not node.args:
                continue
            if isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                keys.add(node.args[0].value)
    for locale in Locale:
        catalog = I18n(locale)._catalog
        assert sorted(keys - set(catalog)) == []


def test_streamlit_and_plotly_runtime_strings_are_catalog_owned():
    project_root = Path(__file__).resolve().parents[1]
    violations = []
    for relative_path in (
        "app.py",
        "meia/preview.py",
        "meia/sidebar.py",
        "meia/viewer.py",
    ):
        path = project_root / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for owner in [tree, *ast.walk(tree)]:
            body = getattr(owner, "body", None)
            if (
                isinstance(body, list)
                and body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and any("\u3400" <= char <= "\u9fff" for char in node.value)
            ):
                violations.append((relative_path, node.lineno, node.value))

    assert violations == []
