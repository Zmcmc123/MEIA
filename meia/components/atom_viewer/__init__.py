"""AtomViewer Streamlit Components v1 Python Adapter。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import streamlit.components.v1 as components

from ...i18n import Locale
from ...view_state import CameraState, camera_for_lattice_axis


_BUILD_DIR = Path(__file__).parent / "frontend" / "dist"
if not (_BUILD_DIR / "index.html").is_file():
    raise RuntimeError(
        "AtomViewer 前端产物缺失；请在 "
        "meia/components/atom_viewer/frontend 运行 npm run build"
    )

_component = components.declare_component(
    "meia_atom_viewer",
    path=str(_BUILD_DIR),
)

_EMPTY_CELL = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)


def render_atom_viewer(
    *,
    figure: Any,
    structure_id: str,
    view_revision: str,
    applied_camera: CameraState,
    selected_atom_index: int | None,
    locale: Locale | str,
    messages: Mapping[str, str],
    key: str,
    axis_cameras: Mapping[str, CameraState] | None = None,
    style_dirty: bool = False,
    selected_atom_indices: Sequence[int] | None = None,
    batch_selection_enabled: bool = False,
    extreme_3d_interaction: bool = False,
) -> Any:
    """渲染 3D 组件并返回其最新用户确认事件。"""
    resolved_locale = Locale(locale).value
    if not isinstance(messages, Mapping) or not messages or not all(
        isinstance(message_key, str) and isinstance(text, str) and text
        for message_key, text in messages.items()
    ):
        raise TypeError("messages must be a non-empty string mapping")
    if not isinstance(extreme_3d_interaction, bool):
        raise TypeError("extreme_3d_interaction must be a boolean")
    # Components v1 使用 stdlib json.dumps；Plotly 的 to_plotly_json 可能
    # 保留 NumPy 数组，因此先通过 Plotly 自身编码器跨越 JSON 边界。
    figure_payload = json.loads(figure.to_json())
    if axis_cameras is None:
        # 保持旧版适配器调用可用；无晶胞信息时回退到笛卡尔正轴。
        axis_cameras = {
            axis: camera_for_lattice_axis(_EMPTY_CELL, axis)
            for axis in ("a", "b", "c")
        }
    return _component(
        figure=figure_payload,
        structure_id=structure_id,
        view_revision=view_revision,
        applied_camera=applied_camera.to_plotly_dict(),
        axis_cameras={
            axis: camera.to_plotly_dict()
            for axis, camera in axis_cameras.items()
        },
        selected_atom_index=selected_atom_index,
        selected_atom_indices=list(selected_atom_indices or ()),
        batch_selection_enabled=bool(batch_selection_enabled),
        extreme_3d_interaction=extreme_3d_interaction,
        style_dirty=bool(style_dirty),
        locale=resolved_locale,
        messages=dict(messages),
        key=key,
        default=None,
    )
