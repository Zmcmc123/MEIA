"""与构型、风格和草稿状态隔离的界面语言会话转换。"""

from __future__ import annotations

import json
from typing import MutableMapping

from .brand import LOCALE_STORAGE_KEY
from .i18n import Locale


APP_LOCALE_KEY = "meia_app_locale"
APP_LOCALE_SOURCE_KEY = "meia_app_locale_source"
APP_LOCALE_WIDGET_KEY = "meia_app_locale_widget"
LOCALE_COOKIE_MAX_AGE_SECONDS = 365 * 24 * 60 * 60


def locale_cookie_markup(locale: Locale | str) -> str:
    """生成无外部静态资源依赖的语言 Cookie 写入脚本。"""
    value = Locale(locale).value
    cookie = (
        f"{LOCALE_STORAGE_KEY}={value}; Path=/; "
        f"Max-Age={LOCALE_COOKIE_MAX_AGE_SECONDS}; SameSite=Lax"
    )
    return f"<script>document.cookie = {json.dumps(cookie)};</script>"


def _locale_from_accept_language(value: object) -> Locale:
    if not isinstance(value, str) or not value.strip():
        return Locale.ZH_CN
    candidates: list[tuple[float, int, str]] = []
    for order, item in enumerate(value.split(",")):
        language, *parameters = item.split(";")
        language = language.strip().lower()
        if not language or language == "*":
            continue
        quality = 1.0
        valid = True
        for parameter in parameters:
            name, separator, raw_value = parameter.partition("=")
            if name.strip().lower() != "q":
                continue
            if not separator:
                valid = False
                break
            try:
                quality = float(raw_value.strip())
            except ValueError:
                valid = False
                break
            if not 0.0 <= quality <= 1.0:
                valid = False
                break
        if valid and quality > 0.0:
            candidates.append((quality, -order, language))
    if not candidates:
        return Locale.ZH_CN
    _, _, browser_language = max(candidates)
    return Locale.ZH_CN if browser_language.startswith("zh") else Locale.EN


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


def initialize_locale(
    session_state: MutableMapping[str, object],
    *,
    stored_locale: object = None,
    accept_language: object = None,
) -> Locale:
    """从已保存偏好初始化与可视化状态隔离的界面语言。"""
    current = load_locale(session_state)
    if current is not None:
        return current
    try:
        resolved = Locale(stored_locale)
    except (TypeError, ValueError):
        resolved = _locale_from_accept_language(accept_language)
        source = "browser"
    else:
        source = "manual"
    session_state[APP_LOCALE_KEY] = resolved.value
    session_state[APP_LOCALE_SOURCE_KEY] = source
    return resolved


def set_manual_locale(
    session_state: MutableMapping[str, object],
    locale: Locale | str,
) -> Locale:
    resolved = Locale(locale)
    session_state[APP_LOCALE_KEY] = resolved.value
    session_state[APP_LOCALE_SOURCE_KEY] = "manual"
    return resolved
