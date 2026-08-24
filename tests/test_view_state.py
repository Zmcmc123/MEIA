"""3D 相机、已应用视角和 Viewer 事件的纯函数测试。"""

import numpy as np
import pytest
from ase.utils import rotate

from meia.view_state import (
    ApplyCameraEvent,
    AtomSelectionBatchEvent,
    AtomSelectionEvent,
    CameraState,
    CameraValidationError,
    ViewerEventError,
    accept_apply_camera_event,
    accept_atom_selection_batch_event,
    accept_atom_selection_event,
    camera_to_rotation_matrix,
    camera_for_lattice_axis,
    parse_apply_camera_event,
    parse_atom_selection_batch_event,
    parse_atom_selection_event,
    rotation_matrix_to_camera,
    structure_id_from_bytes,
)


def valid_event(structure_id="structure-a", event_id="event-1"):
    """返回包含全部协议字段的有效 apply_camera 事件。"""
    return {
        "event_type": "apply_camera",
        "event_id": event_id,
        "structure_id": structure_id,
        "camera": {
            "eye": {"x": 1.0, "y": 1.0, "z": 1.0},
            "up": {"x": 0.0, "y": 0.0, "z": 1.0},
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "projection": {"type": "orthographic"},
        },
    }


def valid_selection_event(
    structure_id="structure-a",
    event_id="selection-1",
    atom_index=1,
    atom_symbol="O",
):
    """返回一个完整的 select_atom 事件。"""
    return {
        "event_type": "select_atom",
        "event_id": event_id,
        "structure_id": structure_id,
        "atom_index": atom_index,
        "atom_symbol": atom_symbol,
    }


def valid_batch_selection_event(
    structure_id="structure-a",
    event_id="selection-batch-1",
    atom_indices=None,
):
    """返回一个完整的 select_atoms 批量确认事件。"""
    return {
        "event_type": "select_atoms",
        "event_id": event_id,
        "structure_id": structure_id,
        "atom_indices": [0, 2] if atom_indices is None else atom_indices,
    }


def test_camera_state_fills_defaults_and_serializes_plotly_shape():
    """缺少字段时仍应得到完整且固定为正交投影的 Plotly 相机。"""
    camera = CameraState.from_mapping({"eye": {"x": 1, "y": 1, "z": 0}})

    assert camera.center == (0.0, 0.0, 0.0)
    assert camera.up == (0.0, 0.0, 1.0)
    assert camera.projection == "orthographic"
    assert camera.to_plotly_dict()["projection"] == {"type": "orthographic"}


def test_camera_matrix_uses_columns_for_row_vector_projection():
    """行向量投影必须依次产生屏幕右、屏幕上和深度坐标。"""
    camera = CameraState(
        eye=(1.0, 1.0, 0.0),
        center=(0.0, 0.0, 0.0),
        up=(0.0, 0.0, 1.0),
    )

    rotation = camera_to_rotation_matrix(camera)
    root_two = np.sqrt(2.0)

    assert np.allclose(
        np.array([-1.0 / root_two, 1.0 / root_two, 0.0]) @ rotation,
        [1.0, 0.0, 0.0],
    )
    assert np.allclose(rotation.T @ rotation, np.eye(3))
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_camera_pan_does_not_change_2d_orientation():
    """Plotly center 平移不属于本功能要同步的视图方向。"""
    applied = CameraState(eye=(1.0, 1.0, 1.0))
    panned = CameraState(
        eye=(1.0, 1.0, 1.0),
        center=(0.3, -0.2, 0.1),
    )

    assert np.allclose(
        camera_to_rotation_matrix(applied),
        camera_to_rotation_matrix(panned),
    )


@pytest.mark.parametrize(
    ("axis", "expected_eye", "expected_up"),
    [
        ("a", (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ("b", (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ("c", (0.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    ],
)
def test_lattice_axis_camera_looks_along_positive_direct_cell_vector(
    axis,
    expected_eye,
    expected_up,
):
    """a/b/c 按钮必须沿对应正晶格矢量观察，且屏幕上方稳定。"""
    camera = camera_for_lattice_axis(
        np.diag([2.0, 3.0, 4.0]),
        axis,
        eye_distance=1.0,
    )

    assert np.allclose(camera.eye, expected_eye)
    assert np.allclose(camera.up, expected_up)
    assert np.dot(camera.eye, camera.up) == pytest.approx(0.0)


def test_lattice_axis_camera_uses_nonorthogonal_direct_vector_and_orthogonalizes_up():
    """非正交晶胞的 a 视角不得退化为全局 X 轴。"""
    cell = np.array(
        [
            [2.0, 1.0, 0.0],
            [0.0, 3.0, 0.5],
            [0.2, 0.0, 4.0],
        ]
    )

    camera = camera_for_lattice_axis(cell, "a", eye_distance=2.0)

    assert np.allclose(
        np.asarray(camera.eye) / np.linalg.norm(camera.eye),
        cell[0] / np.linalg.norm(cell[0]),
    )
    assert np.linalg.norm(camera.eye) == pytest.approx(2.0)
    assert np.linalg.norm(camera.up) == pytest.approx(1.0)
    assert np.dot(camera.eye, camera.up) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "camera",
    [
        {"eye": {"x": 0, "y": 0, "z": 0}},
        {"eye": {"x": 0, "y": 0, "z": 1}, "up": {"x": 0, "y": 0, "z": 2}},
        {"eye": {"x": float("nan"), "y": 1, "z": 1}},
        {"projection": {"type": "perspective"}},
    ],
)
def test_invalid_camera_is_rejected(camera):
    """退化、非有限或透视相机不得覆盖有效 2D 视角。"""
    with pytest.raises(CameraValidationError):
        camera_to_rotation_matrix(camera)


@pytest.mark.parametrize("rotation_text", ["90x", "45z,90x", "54.735x,45z"])
def test_rotation_matrix_camera_round_trip_preserves_orientation(rotation_text):
    """预设矩阵反转为相机后必须无损恢复观察方向。"""
    rotation = rotate(rotation_text)

    camera = rotation_matrix_to_camera(rotation)
    recovered = camera_to_rotation_matrix(camera)

    assert np.allclose(recovered, rotation, atol=1e-10)


@pytest.mark.parametrize(
    "rotation",
    [
        np.eye(2),
        np.full((3, 3), np.nan),
        np.diag([1.0, 1.0, -1.0]),
    ],
)
def test_invalid_rotation_matrix_is_rejected(rotation):
    """非 3×3、非有限或左手矩阵不得生成 Plotly 相机。"""
    with pytest.raises(CameraValidationError):
        rotation_matrix_to_camera(rotation)


def test_structure_id_is_content_sha256():
    """同内容身份稳定，不同内容身份必须不同。"""
    assert structure_id_from_bytes(b"same") == structure_id_from_bytes(b"same")
    assert structure_id_from_bytes(b"same") != structure_id_from_bytes(b"different")
    assert len(structure_id_from_bytes(b"same")) == 64


def test_parse_apply_camera_event_returns_typed_event():
    """原始字典必须在 app 边界外转换为类型化事件。"""
    event = parse_apply_camera_event(valid_event())

    assert isinstance(event, ApplyCameraEvent)
    assert event.event_id == "event-1"
    assert event.camera.projection == "orthographic"


def test_duplicate_event_is_ignored():
    """同一个 event_id 在 Streamlit rerun 时不得再次更新 2D。"""
    accepted = accept_apply_camera_event(
        valid_event(),
        current_structure_id="structure-a",
        processed_event_id="event-1",
    )

    assert accepted is None


def test_stale_structure_event_is_silently_ignored():
    """切换构型后迟到的旧 iframe 事件不得覆盖视角或打扰用户。"""
    accepted = accept_apply_camera_event(
        valid_event(structure_id="old"),
        current_structure_id="new",
        processed_event_id=None,
    )

    assert accepted is None


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("event_id", "event_id"),
        ("structure_id", "structure_id"),
    ],
)
def test_whitespace_only_event_identity_is_rejected(field, message):
    """只含空白的身份不能绕过非空字符串校验。"""
    event = valid_event()
    event[field] = "   "

    with pytest.raises(ViewerEventError, match=message):
        parse_apply_camera_event(event)


def test_none_viewer_event_means_no_user_action():
    """组件尚未提交值时不应被当成错误或相机更新。"""
    assert parse_apply_camera_event(None) is None


@pytest.mark.parametrize(
    "event",
    [
        "apply_camera",
        {"event_type": "selection"},
        {"event_type": "apply_camera", "event_id": "event-1", "structure_id": "s"},
        {
            **valid_event(),
            "camera": {"projection": {"type": "perspective"}},
        },
        {
            **valid_event(),
            "camera": {"eye": {"x": float("nan"), "y": 1.0, "z": 1.0}},
        },
    ],
)
def test_malformed_viewer_event_is_rejected(event):
    """不完整、未知或数值无效的组件事件必须保留上一视角。"""
    with pytest.raises(ViewerEventError):
        parse_apply_camera_event(event)


def test_parse_atom_selection_event_returns_typed_event():
    """原子点选事件应在组件边界转换为带类型的零基索引。"""
    event = parse_atom_selection_event(valid_selection_event())

    assert isinstance(event, AtomSelectionEvent)
    assert event.atom_index == 1
    assert event.atom_symbol == "O"


def test_atom_selection_accepts_matching_current_structure_and_symbol():
    """只有当前构型中索引和元素均匹配的点选才可更新选择。"""
    event = accept_atom_selection_event(
        valid_selection_event(),
        current_structure_id="structure-a",
        processed_event_id=None,
        atom_symbols=["C", "O", "H"],
    )

    assert event == AtomSelectionEvent(
        event_id="selection-1",
        structure_id="structure-a",
        atom_index=1,
        atom_symbol="O",
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_structure_id": "new"},
        {"processed_event_id": "selection-1"},
    ],
)
def test_stale_or_duplicate_atom_selection_is_ignored(kwargs):
    """旧 iframe 事件和 Streamlit 重放事件都不应重复选择原子。"""
    arguments = {
        "current_structure_id": "structure-a",
        "processed_event_id": None,
        "atom_symbols": ["C", "O"],
    }
    arguments.update(kwargs)

    assert accept_atom_selection_event(valid_selection_event(), **arguments) is None


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (valid_selection_event(atom_index=True), "atom_index"),
        (valid_selection_event(atom_index=1.5), "atom_index"),
        (valid_selection_event(atom_symbol=""), "atom_symbol"),
        ({**valid_selection_event(), "event_id": " "}, "event_id"),
        ({**valid_selection_event(), "structure_id": " "}, "structure_id"),
    ],
)
def test_malformed_atom_selection_is_rejected(event, message):
    """组件不能用隐式数值转换或空身份绕过点选协议。"""
    with pytest.raises(ViewerEventError, match=message):
        parse_atom_selection_event(event)


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (valid_selection_event(atom_index=-1), "range"),
        (valid_selection_event(atom_index=2), "range"),
        (valid_selection_event(atom_symbol="C"), "symbol"),
    ],
)
def test_atom_selection_must_match_current_atom_table(event, message):
    """越界或索引元素不一致说明事件与当前结构不可信。"""
    with pytest.raises(ViewerEventError, match=message):
        accept_atom_selection_event(
            event,
            current_structure_id="structure-a",
            processed_event_id=None,
            atom_symbols=["C", "O"],
        )


def test_parse_atom_selection_batch_event_returns_complete_canonical_set():
    """前端确认后必须一次携带完整、规范化的零基索引集合。"""
    event = parse_atom_selection_batch_event(valid_batch_selection_event())

    assert event == AtomSelectionBatchEvent(
        event_id="selection-batch-1",
        structure_id="structure-a",
        atom_indices=(0, 2),
    )


def test_atom_selection_batch_accepts_current_structure_once():
    """批量选区只在结构匹配且未处理过时进入 Streamlit 状态。"""
    event = accept_atom_selection_batch_event(
        valid_batch_selection_event(),
        current_structure_id="structure-a",
        processed_event_id=None,
        atom_count=3,
    )

    assert event is not None
    assert event.atom_indices == (0, 2)
    assert accept_atom_selection_batch_event(
        valid_batch_selection_event(),
        current_structure_id="structure-a",
        processed_event_id="selection-batch-1",
        atom_count=3,
    ) is None


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (valid_batch_selection_event(atom_indices=[0, True]), "atom_indices"),
        (valid_batch_selection_event(atom_indices=[2, 0]), "canonical"),
        (valid_batch_selection_event(atom_indices=[0, 0]), "canonical"),
        (valid_batch_selection_event(atom_indices="0,2"), "atom_indices"),
    ],
)
def test_malformed_atom_selection_batch_is_rejected(event, message):
    """批量协议拒绝隐式转换、重复和非升序索引。"""
    with pytest.raises(ViewerEventError, match=message):
        parse_atom_selection_batch_event(event)


def test_atom_selection_batch_rejects_out_of_range_index():
    """当前构型之外的索引不能进入批量选择状态。"""
    with pytest.raises(ViewerEventError, match="range"):
        accept_atom_selection_batch_event(
            valid_batch_selection_event(atom_indices=[0, 3]),
            current_structure_id="structure-a",
            processed_event_id=None,
            atom_count=3,
        )
