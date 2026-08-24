"""MEIA 的集中式翻译、复数格式化与本地化错误接缝。"""

from __future__ import annotations

from enum import Enum
from importlib.resources import files
import json
from string import Formatter
from types import MappingProxyType
from typing import Mapping


class Locale(str, Enum):
    """MEIA 当前支持的界面语言。"""

    ZH_CN = "zh-CN"
    EN = "en"


class LocalizedError(ValueError):
    """同时保留技术诊断与稳定翻译键的领域错误。"""

    def __init__(
        self,
        technical_message: str,
        *,
        message_key: str | None = None,
        message_params: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(technical_message)
        self.message_key = message_key
        self.message_params = MappingProxyType(dict(message_params or {}))


def _load_catalog(locale: Locale) -> Mapping[str, str]:
    resource = files("meia.locales").joinpath(f"{locale.value}.json")
    value = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(text, str)
        for key, text in value.items()
    ):
        raise ValueError(f"invalid locale catalog: {locale.value}")
    return MappingProxyType(dict(value))


_CATALOGS = MappingProxyType({locale: _load_catalog(locale) for locale in Locale})


def _placeholder_names(template: str) -> frozenset[str]:
    return frozenset(
        field_name
        for _, field_name, _, _ in Formatter().parse(template)
        if field_name is not None
    )


def validate_catalog_pair() -> None:
    """确保两份词典键集合和格式化参数完全相同。"""
    chinese = _CATALOGS[Locale.ZH_CN]
    english = _CATALOGS[Locale.EN]
    if set(chinese) != set(english):
        missing_en = sorted(set(chinese) - set(english))
        missing_zh = sorted(set(english) - set(chinese))
        raise ValueError(
            f"locale key mismatch: missing_en={missing_en}, missing_zh={missing_zh}"
        )
    mismatches = [
        key
        for key in chinese
        if _placeholder_names(chinese[key]) != _placeholder_names(english[key])
    ]
    if mismatches:
        raise ValueError(f"locale placeholder mismatch: {sorted(mismatches)}")


class I18n:
    """按显式 locale 读取和格式化 MEIA 用户可见词条。"""

    def __init__(self, locale: Locale | str) -> None:
        self.locale = Locale(locale)
        self._catalog = _CATALOGS[self.locale]

    def text(self, key: str, **params: object) -> str:
        resolved_key = key
        if resolved_key not in self._catalog and "count" in params:
            if self.locale is Locale.EN and params["count"] == 1:
                resolved_key = f"{key}.one"
            else:
                resolved_key = f"{key}.other"
        try:
            template = self._catalog[resolved_key]
        except KeyError as exc:
            raise KeyError(key) from exc
        return template.format(**params)

    def bundle(self, prefix: str) -> dict[str, str]:
        namespace = prefix.rstrip(".") + "."
        return {
            key[len(namespace) :]: value
            for key, value in self._catalog.items()
            if key.startswith(namespace)
        }

    def error_text(self, error: BaseException, fallback_key: str) -> str:
        if isinstance(error, LocalizedError) and error.message_key:
            return self.text(error.message_key, **error.message_params)
        detail = f"{type(error).__name__}: {error}"
        return self.text(fallback_key, detail=detail)


validate_catalog_pair()
