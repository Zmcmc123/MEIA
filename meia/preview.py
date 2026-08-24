"""Streamlit 2D 预览的固定窗口与高像素密度输出。"""

from __future__ import annotations

import base64
from html import escape
import io

import matplotlib.pyplot as plt
import numpy as np


PREVIEW_CSS_WIDTH = 900
PREVIEW_CSS_HEIGHT = 675
PREVIEW_PIXEL_WIDTH = 1800
PREVIEW_PIXEL_HEIGHT = 1350


def preview_image_html(payload: bytes, *, alt_text: str) -> str:
    """把 2× PNG 嵌入固定 CSS 窗口，避免 Streamlit 二次降采样。"""
    if not isinstance(alt_text, str) or not alt_text.strip():
        raise ValueError("alt_text must be a non-empty string")
    encoded = base64.b64encode(payload).decode("ascii")
    return (
        '<img class="meia-2d-preview" '
        f'src="data:image/png;base64,{encoded}" '
        f'alt="{escape(alt_text.strip(), quote=True)}" '
        f'width="{PREVIEW_CSS_WIDTH}" height="{PREVIEW_CSS_HEIGHT}" '
        f'style="display:block;width:{PREVIEW_CSS_WIDTH}px;'
        f'height:{PREVIEW_CSS_HEIGHT}px;max-width:none;" />'
    )


def render_preview_png(
    fig: plt.Figure,
    *,
    transparent: bool,
) -> bytes:
    """生成与 900×675 CSS 窗口对应的 2× PNG 预览。"""
    width_inches, height_inches = fig.get_size_inches()
    dpi_x = PREVIEW_PIXEL_WIDTH / float(width_inches)
    dpi_y = PREVIEW_PIXEL_HEIGHT / float(height_inches)
    if not np.isclose(dpi_x, dpi_y, atol=1e-12):
        raise ValueError("2D preview figure must keep a 4:3 aspect ratio")

    buffer = io.BytesIO()
    fig.savefig(
        buffer,
        format="png",
        dpi=dpi_x,
        transparent=transparent,
        bbox_inches=None,
        pad_inches=0,
    )
    return buffer.getvalue()
