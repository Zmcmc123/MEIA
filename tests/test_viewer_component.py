"""AtomViewer Python Adapter 与已应用视角状态测试。"""

from unittest.mock import Mock

import numpy as np
from ase import Atoms
from ase.utils import rotate

from meia.components.atom_viewer import render_atom_viewer
from meia.config import RenderConfig
from meia.i18n import I18n, Locale
from meia.view_state import (
    AcceptedCameraEvent,
    CameraState,
    camera_to_rotation_matrix,
    initial_applied_view,
    load_applied_view_state,
    store_applied_view_state,
    update_applied_view,
)
from meia.viewer import create_3d_figure as _create_3d_figure


def test_nonfinite_camera_has_exact_english_diagnostic():
    with np.testing.assert_raises(ValueError) as captured:
        CameraState(eye=(np.inf, 0.0, 0.0))

    assert I18n(Locale.EN).error_text(
        captured.exception, "viewer.event_failed"
    ) == "The camera is invalid: camera eye must be a finite 3-vector"


def create_3d_figure(*args, **kwargs):
    kwargs.setdefault(
        "figure_messages",
        I18n(Locale.ZH_CN).bundle("figure3d"),
    )
    return _create_3d_figure(*args, **kwargs)


def test_component_adapter_crosses_plotly_json_boundary(monkeypatch):
    """Adapter 必须传入标准 JSON 标量，避免 NumPy 数组破坏组件序列化。"""
    component = Mock(return_value=None)
    monkeypatch.setattr("meia.components.atom_viewer._component", component)
    figure = Mock()
    figure.to_json.return_value = '{"data": [], "layout": {}}'

    result = render_atom_viewer(
        figure=figure,
        structure_id="abc",
        view_revision="preset:abc:90x",
        applied_camera=CameraState(),
        axis_cameras={
            "a": CameraState(eye=(2.0, 0.0, 0.0)),
            "b": CameraState(eye=(0.0, 2.0, 0.0)),
            "c": CameraState(eye=(0.0, 0.0, 2.0), up=(0.0, 1.0, 0.0)),
        },
        selected_atom_index=None,
        selected_atom_indices=(0, 2),
        batch_selection_enabled=True,
        locale=Locale.EN,
        messages=I18n(Locale.EN).bundle("viewer"),
        key="viewer",
        style_dirty=True,
    )

    assert result is None
    component.assert_called_once_with(
        figure={"data": [], "layout": {}},
        structure_id="abc",
        view_revision="preset:abc:90x",
        applied_camera=CameraState().to_plotly_dict(),
        axis_cameras={
            "a": CameraState(eye=(2.0, 0.0, 0.0)).to_plotly_dict(),
            "b": CameraState(eye=(0.0, 2.0, 0.0)).to_plotly_dict(),
            "c": CameraState(
                eye=(0.0, 0.0, 2.0),
                up=(0.0, 1.0, 0.0),
            ).to_plotly_dict(),
        },
        selected_atom_index=None,
        selected_atom_indices=[0, 2],
        batch_selection_enabled=True,
        locale="en",
        messages=I18n(Locale.EN).bundle("viewer"),
        style_dirty=True,
        key="viewer",
        default=None,
    )


def test_component_adapter_keeps_legacy_callers_working_without_axis_cameras(
    monkeypatch,
):
    """新增晶轴工具不应把原有 Python 适配器调用改成必填参数。"""
    component = Mock(return_value=None)
    monkeypatch.setattr("meia.components.atom_viewer._component", component)
    figure = Mock()
    figure.to_json.return_value = '{"data": [], "layout": {}}'

    render_atom_viewer(
        figure=figure,
        structure_id="legacy",
        view_revision="legacy-view",
        applied_camera=CameraState(),
        selected_atom_index=None,
        locale=Locale.ZH_CN,
        messages=I18n(Locale.ZH_CN).bundle("viewer"),
        key="viewer",
    )

    axis_cameras = component.call_args.kwargs["axis_cameras"]
    assert component.call_args.kwargs["style_dirty"] is False
    assert component.call_args.kwargs["selected_atom_indices"] == []
    assert component.call_args.kwargs["batch_selection_enabled"] is False
    assert set(axis_cameras) == {"a", "b", "c"}
    assert axis_cameras["a"]["eye"]["x"] > 0
    assert axis_cameras["b"]["eye"]["y"] > 0
    assert axis_cameras["c"]["eye"]["z"] > 0
    assert axis_cameras["c"]["up"] == {"x": 0.0, "y": 1.0, "z": 0.0}


def test_component_adapter_forwards_locale_and_viewer_messages(monkeypatch):
    component = Mock(return_value=None)
    monkeypatch.setattr("meia.components.atom_viewer._component", component)
    figure = Mock()
    figure.to_json.return_value = '{"data": [], "layout": {}}'
    messages = I18n(Locale.EN).bundle("viewer")

    render_atom_viewer(
        figure=figure,
        structure_id="structure",
        view_revision="revision",
        applied_camera=CameraState(),
        selected_atom_index=None,
        locale=Locale.EN,
        messages=messages,
        key="viewer",
    )

    assert component.call_args.kwargs["locale"] == "en"
    assert component.call_args.kwargs["messages"]["camera.apply"] == (
        "Apply Current View"
    )


def test_english_3d_figure_uses_detected_bond_terminology():
    atoms = Atoms(
        "OH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )

    figure = create_3d_figure(
        atoms,
        RenderConfig(show_unit_cell=1),
        figure_messages=I18n(Locale.EN).bundle("figure3d"),
    )

    names = {trace.name for trace in figure.data}
    assert "Atoms" in names
    assert "Detected Bonds" in names
    assert "Unit Cell" in names
    assert not any("Covalent" in name for name in names)
    atom_trace = next(trace for trace in figure.data if trace.name == "Atoms")
    assert "image" in atom_trace.text[0]


def test_initial_applied_view_uses_rotation_preset():
    """预设必须同时生成完全一致的 3D 相机和 2D 旋转矩阵。"""
    state = initial_applied_view(
        "90x",
        view_revision="preset:structure-a:90x",
    )

    assert np.allclose(state.rotation_matrix, rotate("90x"))
    assert np.allclose(camera_to_rotation_matrix(state.camera), state.rotation_matrix)
    assert state.view_revision == "preset:structure-a:90x"


def test_accepted_event_replaces_applied_view_atomically():
    """一次用户确认必须同步替换相机、矩阵、事件 ID 和 revision。"""
    initial = initial_applied_view(
        "90x",
        view_revision="preset:structure-a:90x",
    )
    camera = CameraState(eye=(0.0, 2.0, 0.0))
    accepted = AcceptedCameraEvent(
        event_id="event-2",
        camera=camera,
        rotation_matrix=camera_to_rotation_matrix(camera),
    )

    updated = update_applied_view(initial, accepted)

    assert updated.event_id == "event-2"
    assert updated.camera == camera
    assert np.allclose(updated.rotation_matrix, accepted.rotation_matrix)
    assert updated.view_revision == "camera:event-2"


def test_applied_view_session_adapter_uses_fixed_keys():
    """Session 适配器必须集中维护设计中约定的状态键。"""
    session = {}
    expected = initial_applied_view(
        "90x",
        view_revision="preset:structure-a:90x",
    )

    store_applied_view_state(session, expected)
    actual = load_applied_view_state(session)

    assert set(session) == {
        "meia_applied_camera",
        "meia_applied_rotation_matrix",
        "meia_processed_viewer_event_id",
        "meia_view_revision",
    }
    assert actual.camera == expected.camera
    assert np.allclose(actual.rotation_matrix, expected.rotation_matrix)
    assert actual.event_id is None


def test_component_is_declared_in_meia_namespace():
    import inspect
    import meia.components.atom_viewer as component_module

    source = inspect.getsource(component_module)
    assert '"meia_atom_viewer"' in source
