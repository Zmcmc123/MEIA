"""浏览器语言偏好组件的 Python 边界测试。"""

from __future__ import annotations

import pytest

import meia.components.locale_preference as locale_module
from meia.i18n import Locale
from meia.components.locale_preference import LocalePreference


def test_locale_preference_adapter_validates_component_result(monkeypatch):
    component_calls = []

    def component(**kwargs):
        component_calls.append(kwargs)
        return {"locale": "en", "source": "browser"}

    monkeypatch.setattr(locale_module, "_component", component)
    result = locale_module.render_locale_preference(
        persist_locale=None,
        key="locale",
    )

    assert result == LocalePreference(Locale.EN, "browser")
    assert component_calls == [
        {
            "persist_locale": None,
            "storage_key": "meia.locale",
            "key": "locale",
            "default": None,
        }
    ]


@pytest.mark.parametrize(
    "payload",
    [
        {"locale": "fr", "source": "browser"},
        {"locale": "en", "source": "cookie"},
        "en",
    ],
)
def test_locale_preference_adapter_rejects_malformed_results(monkeypatch, payload):
    monkeypatch.setattr(locale_module, "_component", lambda **_kwargs: payload)
    with pytest.raises(ValueError, match="locale preference"):
        locale_module.render_locale_preference(persist_locale=None, key="locale")


def test_locale_preference_adapter_serializes_manual_locale(monkeypatch):
    calls = []
    monkeypatch.setattr(
        locale_module,
        "_component",
        lambda **kwargs: calls.append(kwargs) or None,
    )
    assert locale_module.render_locale_preference(
        persist_locale=Locale.ZH_CN,
        key="locale",
    ) is None
    assert calls[0]["persist_locale"] == "zh-CN"
