"""不可见的浏览器语言偏好 Streamlit 组件。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import streamlit.components.v1 as components

from ...brand import LOCALE_STORAGE_KEY
from ...i18n import Locale


_BUILD_DIR = (
    Path(__file__).parents[1]
    / "atom_viewer"
    / "frontend"
    / "dist"
    / "locale"
)
if not (_BUILD_DIR / "index.html").is_file():
    raise RuntimeError(
        "MEIA locale preference 前端产物缺失；请在 "
        "meia/components/atom_viewer/frontend 运行 npm run build"
    )

_component = components.declare_component(
    "meia_locale_preference",
    path=str(_BUILD_DIR),
)


@dataclass(frozen=True)
class LocalePreference:
    locale: Locale
    source: Literal["stored", "browser"]


def render_locale_preference(
    *,
    persist_locale: Locale | None,
    key: str,
) -> LocalePreference | None:
    value = _component(
        persist_locale=None if persist_locale is None else persist_locale.value,
        storage_key=LOCALE_STORAGE_KEY,
        key=key,
        default=None,
    )
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("locale preference component returned an invalid value")
    try:
        locale = Locale(value.get("locale"))
    except (TypeError, ValueError) as exc:
        raise ValueError("locale preference component returned an invalid locale") from exc
    source = value.get("source")
    if source not in {"stored", "browser"}:
        raise ValueError("locale preference component returned an invalid source")
    return LocalePreference(locale, source)
