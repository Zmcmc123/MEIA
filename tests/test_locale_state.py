"""界面语言会话状态不应改写可视化状态。"""

from __future__ import annotations

from meia.components.locale_preference import LocalePreference
from meia.i18n import Locale
from meia.locale_state import (
    APP_LOCALE_KEY,
    APP_LOCALE_SOURCE_KEY,
    accept_locale_preference,
    load_locale,
    set_manual_locale,
)


def test_stored_preference_initializes_locale_without_visual_state_keys():
    visual_state = object()
    session = {"meia_visual_state": visual_state}
    preference = LocalePreference(Locale.EN, "stored")

    assert accept_locale_preference(session, preference) is Locale.EN
    assert session[APP_LOCALE_KEY] == "en"
    assert session[APP_LOCALE_SOURCE_KEY] == "manual"
    assert session["meia_visual_state"] is visual_state


def test_manual_locale_wins_over_later_browser_event():
    session = {}
    set_manual_locale(session, Locale.EN)
    result = accept_locale_preference(
        session,
        LocalePreference(Locale.ZH_CN, "browser"),
    )
    assert result is Locale.EN


def test_invalid_session_locale_is_removed_and_reinitialized():
    session = {APP_LOCALE_KEY: "fr"}
    assert load_locale(session) is None
    assert APP_LOCALE_KEY not in session


def test_browser_preference_remains_browser_until_manual_choice():
    session = {}
    assert accept_locale_preference(
        session,
        LocalePreference(Locale.EN, "browser"),
    ) is Locale.EN
    assert session[APP_LOCALE_SOURCE_KEY] == "browser"

    assert set_manual_locale(session, Locale.ZH_CN) is Locale.ZH_CN
    assert session[APP_LOCALE_SOURCE_KEY] == "manual"
