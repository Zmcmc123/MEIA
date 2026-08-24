"""MEIA 品牌、版本与 JSON 命名契约。"""

from __future__ import annotations

import json

import pytest

from meia.brand import (
    DEFAULT_EXPORT_STEM,
    JSON_EXTENSION,
    LOCALE_STORAGE_KEY,
    MEIA_VERSION,
    PRODUCT_FULL_NAME,
    PRODUCT_NAME,
    SESSION_KEY_PREFIX,
    STYLE_JSON_SUFFIX,
    WORKSPACE_JSON_SUFFIX,
    session_key,
)
from meia.presets import (
    PresetError,
    load_default_style,
    parse_preset,
    style_preset_to_json,
)


def test_meia_brand_constants_are_the_single_naming_contract():
    assert PRODUCT_NAME == "MEIA"
    assert (
        PRODUCT_FULL_NAME
        == "Molecular and Extended-system Illustration Assistant"
    )
    assert MEIA_VERSION == "0.11.0"
    assert SESSION_KEY_PREFIX == "meia_"
    assert LOCALE_STORAGE_KEY == "meia.locale"
    assert JSON_EXTENSION == ".meia.json"
    assert STYLE_JSON_SUFFIX == ".style.meia.json"
    assert WORKSPACE_JSON_SUFFIX == ".workspace.meia.json"
    assert DEFAULT_EXPORT_STEM == "meia-visual-state"
    assert session_key("visual_state") == "meia_visual_state"


@pytest.mark.parametrize("value", ["", None, 1, False])
def test_session_key_rejects_invalid_suffixes(value):
    with pytest.raises(ValueError, match="non-empty string"):
        session_key(value)  # type: ignore[arg-type]


def test_default_style_uses_meia_v7_metadata_only():
    payload = json.loads(style_preset_to_json(load_default_style()))
    assert payload["schema_version"] == 7
    assert payload["meia_version"] == "0.11.0"


def test_unknown_metadata_is_rejected():
    payload = json.loads(style_preset_to_json(load_default_style()))
    payload["unsupported_metadata"] = "legacy"
    with pytest.raises(PresetError, match="未知字段"):
        parse_preset(json.dumps(payload))
