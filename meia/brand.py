"""MEIA 产品与持久化命名的唯一 Python 接缝。"""

from __future__ import annotations


PRODUCT_NAME = "MEIA"
PRODUCT_FULL_NAME = "Molecular and Extended-system Illustration Assistant"
MEIA_VERSION = "0.11.0"
SESSION_KEY_PREFIX = "meia_"
LOCALE_STORAGE_KEY = "meia.locale"
JSON_EXTENSION = ".meia.json"
STYLE_JSON_SUFFIX = ".style.meia.json"
WORKSPACE_JSON_SUFFIX = ".workspace.meia.json"
DEFAULT_EXPORT_STEM = "meia-visual-state"


def session_key(suffix: str) -> str:
    """为 MEIA 会话状态和组件生成稳定键。"""
    if not isinstance(suffix, str) or not suffix:
        raise ValueError("session key suffix must be a non-empty string")
    return f"{SESSION_KEY_PREFIX}{suffix}"
