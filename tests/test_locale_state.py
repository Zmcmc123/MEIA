"""界面语言会话状态不应改写可视化状态。"""

from __future__ import annotations

import pytest

import meia.locale_state as locale_state
from meia.i18n import Locale
from meia.locale_state import (
    APP_LOCALE_KEY,
    APP_LOCALE_SOURCE_KEY,
    load_locale,
    set_manual_locale,
)


def test_stored_preference_initializes_locale_without_visual_state_keys():
    visual_state = object()
    session = {"meia_visual_state": visual_state}

    assert locale_state.initialize_locale(
        session,
        stored_locale="en",
        accept_language="zh-CN,zh;q=0.9",
    ) is Locale.EN
    assert session[APP_LOCALE_KEY] == "en"
    assert session[APP_LOCALE_SOURCE_KEY] == "manual"
    assert session["meia_visual_state"] is visual_state


@pytest.mark.parametrize(
    ("accept_language", "expected"),
    [
        ("en-US,en;q=0.9", Locale.EN),
        ("zh-TW,zh;q=0.9,en;q=0.8", Locale.ZH_CN),
        ("fr-FR,fr;q=0.9", Locale.EN),
        ("en;q=0,zh-CN;q=1", Locale.ZH_CN),
        ("zh-CN;q=0.2,en;q=0.9", Locale.EN),
        (None, Locale.ZH_CN),
        ("", Locale.ZH_CN),
    ],
)
def test_browser_language_initializes_locale_with_chinese_fallback(
    accept_language,
    expected,
):
    session = {}

    assert locale_state.initialize_locale(
        session,
        stored_locale=None,
        accept_language=accept_language,
    ) is expected
    assert session[APP_LOCALE_SOURCE_KEY] == "browser"


def test_manual_locale_cookie_markup_persists_for_the_whole_app():
    assert locale_state.locale_cookie_markup(Locale.EN) == (
        '<script>document.cookie = "meia.locale=en; Path=/; '
        'Max-Age=31536000; SameSite=Lax";</script>'
    )


def test_manual_locale_wins_over_later_browser_event():
    session = {}
    set_manual_locale(session, Locale.EN)
    result = locale_state.initialize_locale(
        session,
        stored_locale=None,
        accept_language="zh-CN,zh;q=0.9",
    )
    assert result is Locale.EN


def test_invalid_session_locale_is_removed_and_reinitialized():
    session = {APP_LOCALE_KEY: "fr"}
    assert load_locale(session) is None
    assert APP_LOCALE_KEY not in session


def test_browser_preference_remains_browser_until_manual_choice():
    session = {}
    assert locale_state.initialize_locale(
        session,
        stored_locale=None,
        accept_language="en-US,en;q=0.9",
    ) is Locale.EN
    assert session[APP_LOCALE_SOURCE_KEY] == "browser"

    assert set_manual_locale(session, Locale.ZH_CN) is Locale.ZH_CN
    assert session[APP_LOCALE_SOURCE_KEY] == "manual"
