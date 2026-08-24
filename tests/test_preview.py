"""Streamlit 2D 高清预览的固定像素契约。"""

from io import BytesIO

from PIL import Image

from meia.config import RenderConfig
from meia.i18n import I18n, Locale
from meia.preview import (
    PREVIEW_CSS_HEIGHT,
    PREVIEW_CSS_WIDTH,
    PREVIEW_PIXEL_HEIGHT,
    PREVIEW_PIXEL_WIDTH,
    preview_image_html,
    render_preview_png,
)
from meia.view import render_2d


def test_preview_is_exactly_two_device_pixels_per_css_pixel(sample_atoms):
    """页面窗口应固定为 900×675，实际位图为两倍分辨率。"""
    fig = render_2d(sample_atoms, RenderConfig(dpi=600, show_unit_cell=0))

    payload = render_preview_png(fig, transparent=True)

    with Image.open(BytesIO(payload)) as image:
        assert image.format == "PNG"
        assert image.size == (1800, 1350)
    assert (PREVIEW_CSS_WIDTH, PREVIEW_CSS_HEIGHT) == (900, 675)
    assert (PREVIEW_PIXEL_WIDTH, PREVIEW_PIXEL_HEIGHT) == (1800, 1350)

    import matplotlib.pyplot as plt

    plt.close(fig)


def test_preview_html_preserves_two_x_source_without_streamlit_resampling():
    """HTML 只缩放显示尺寸，不改写 2× PNG 的内部像素。"""
    payload = b"synthetic-png"

    markup = preview_image_html(
        payload,
        alt_text=I18n(Locale.EN).text("preview.alt"),
    )

    assert 'width="900"' in markup
    assert 'height="675"' in markup
    assert "data:image/png;base64,c3ludGhldGljLXBuZw==" in markup
    assert "meia-2d-preview" in markup
    assert 'alt="Flattened 2D atomic-structure preview"' in markup
