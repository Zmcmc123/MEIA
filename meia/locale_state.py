"""与构型、风格和草稿状态隔离的界面语言会话转换。"""

from __future__ import annotations

from typing import MutableMapping

from .components.locale_preference import LocalePreference
from .i18n import Locale


APP_LOCALE_KEY = "meia_app_locale"
APP_LOCALE_SOURCE_KEY = "meia_app_locale_source"
APP_LOCALE_WIDGET_KEY = "meia_app_locale_widget"


def load_locale(session_state: MutableMapping[str, object]) -> Locale | None:
    value = session_state.get(APP_LOCALE_KEY)
    if value is None:
        return None
    try:
        return Locale(value)
    except (TypeError, ValueError):
        session_state.pop(APP_LOCALE_KEY, None)
        session_state.pop(APP_LOCALE_SOURCE_KEY, None)
        return None


def accept_locale_preference(
    session_state: MutableMapping[str, object],
    preference: LocalePreference,
) -> Locale:
    current = load_locale(session_state)
    if current is not None and session_state.get(APP_LOCALE_SOURCE_KEY) == "manual":
        return current
    session_state[APP_LOCALE_KEY] = preference.locale.value
    session_state[APP_LOCALE_SOURCE_KEY] = (
        "manual" if preference.source == "stored" else "browser"
    )
    return preference.locale


def set_manual_locale(
    session_state: MutableMapping[str, object],
    locale: Locale | str,
) -> Locale:
    resolved = Locale(locale)
    session_state[APP_LOCALE_KEY] = resolved.value
    session_state[APP_LOCALE_SOURCE_KEY] = "manual"
    return resolved
