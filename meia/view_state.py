"""3D Viewer 相机、已应用视角和事件边界的纯函数。"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from hashlib import sha256
from math import sqrt
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from .i18n import LocalizedError
from ase.utils import rotate


Vector3 = tuple[float, float, float]
LatticeAxis = Literal["a", "b", "c"]
DEFAULT_EYE_DISTANCE = sqrt(3 * 1.25**2)


class CameraValidationError(LocalizedError):
    """Plotly 相机无法安全转换为正交旋转时抛出。"""

    def __init__(self, technical_message: str) -> None:
        super().__init__(
            technical_message,
            message_key="camera.invalid",
            message_params={"detail": technical_message},
        )


class ViewerEventError(LocalizedError):
    """Viewer 事件格式、结构身份或相机值无效时抛出。"""

    def __init__(self, technical_message: str) -> None:
        super().__init__(
            technical_message,
            message_key="viewer.invalid_event",
            message_params={"detail": technical_message},
        )


def _vector_from_mapping(
    value: Mapping[str, Any],
    name: str,
    default: Vector3,
) -> Vector3:
    raw = value.get(name, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise CameraValidationError(f"camera {name} must be a mapping")

    result = []
    for axis, fallback in zip(("x", "y", "z"), default):
        try:
            component = float(raw.get(axis, fallback))
        except (TypeError, ValueError) as exc:
            raise CameraValidationError(
                f"camera {name}.{axis} must be numeric"
            ) from exc
        if not np.isfinite(component):
            raise CameraValidationError(f"camera {name}.{axis} must be finite")
        result.append(component)
    return tuple(result)  # type: ignore[return-value]


@dataclass(frozen=True)
class CameraState:
    """与 Plotly scene.camera 对应的规范化正交相机。"""

    eye: Vector3 = (1.25, 1.25, 1.25)
    up: Vector3 = (0.0, 0.0, 1.0)
    center: Vector3 = (0.0, 0.0, 0.0)
    projection: Literal["orthographic"] = "orthographic"

    def __post_init__(self) -> None:
        if self.projection != "orthographic":
            raise CameraValidationError("camera projection must be orthographic")
        for name in ("eye", "up", "center"):
            try:
                vector = np.asarray(getattr(self, name), dtype=float)
            except (TypeError, ValueError) as exc:
                raise CameraValidationError(
                    f"camera {name} must be a finite 3-vector"
                ) from exc
            if vector.shape != (3,) or not np.isfinite(vector).all():
                raise CameraValidationError(
                    f"camera {name} must be a finite 3-vector"
                )
            object.__setattr__(self, name, tuple(float(x) for x in vector))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "CameraState":
        if not isinstance(value, Mapping):
            raise CameraValidationError("camera must be a mapping")
        projection_value = value.get("projection", "orthographic")
        if isinstance(projection_value, Mapping):
            projection_value = projection_value.get("type", "orthographic")
        if projection_value != "orthographic":
            raise CameraValidationError("camera projection must be orthographic")
        return cls(
            eye=_vector_from_mapping(value, "eye", cls.eye),
            up=_vector_from_mapping(value, "up", cls.up),
            center=_vector_from_mapping(value, "center", cls.center),
        )

    def to_plotly_dict(self) -> dict[str, Any]:
        """返回仅含 JSON 标量的 Plotly scene.camera 字典。"""
        return {
            "eye": dict(zip(("x", "y", "z"), self.eye)),
            "up": dict(zip(("x", "y", "z"), self.up)),
            "center": dict(zip(("x", "y", "z"), self.center)),
            "projection": {"type": self.projection},
        }


def camera_to_rotation_matrix(
    camera: CameraState | Mapping[str, Any],
) -> np.ndarray:
    """把 Plotly 相机转换成适用于 ``positions @ matrix`` 的旋转矩阵。"""
    state = camera if isinstance(camera, CameraState) else CameraState.from_mapping(camera)
    eye = np.asarray(state.eye, dtype=float)
    up = np.asarray(state.up, dtype=float)

    # Plotly 的 eye 是相对 center 的相机向量；center 只表示正交视图平移。
    # 本功能明确只同步方向，因此不让平移改变 2D 旋转矩阵。
    view = -eye
    if np.linalg.norm(view) <= 1e-12:
        raise CameraValidationError("camera eye vector must be non-zero")
    view /= np.linalg.norm(view)

    right = np.cross(view, up)
    if np.linalg.norm(right) <= 1e-12:
        raise CameraValidationError("camera up must not be parallel to view")
    right /= np.linalg.norm(right)

    corrected_up = np.cross(right, view)
    corrected_up /= np.linalg.norm(corrected_up)
    rotation = np.column_stack((right, corrected_up, -view))

    if not np.isfinite(rotation).all() or not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-10,
    ):
        raise CameraValidationError("camera produced an invalid rotation matrix")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10):
        raise CameraValidationError("camera rotation must be right-handed")
    return rotation


def rotation_matrix_to_camera(
    rotation_matrix: np.ndarray,
    eye_distance: float = DEFAULT_EYE_DISTANCE,
) -> CameraState:
    """把 MEIA 行向量旋转矩阵反向转换为 Plotly 正交相机。"""
    rotation = np.asarray(rotation_matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise CameraValidationError("rotation matrix must be a finite 3x3 array")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-10):
        raise CameraValidationError("rotation matrix must be orthogonal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-10):
        raise CameraValidationError("rotation matrix must be right-handed")
    if not np.isfinite(eye_distance) or eye_distance <= 0:
        raise CameraValidationError("eye distance must be finite and positive")

    depth = rotation[:, 2]
    up = rotation[:, 1]
    return CameraState(
        eye=tuple(float(x) for x in depth * eye_distance),
        up=tuple(float(x) for x in up),
        center=(0.0, 0.0, 0.0),
    )


def camera_for_lattice_axis(
    cell: Sequence[Sequence[float]],
    axis: LatticeAxis,
    *,
    eye_distance: float = DEFAULT_EYE_DISTANCE,
) -> CameraState:
    """在正晶格轴方向放置相机，并构造稳定的屏幕上方。"""
    axis_indices = {"a": 0, "b": 1, "c": 2}
    if axis not in axis_indices:
        raise CameraValidationError(f"unsupported lattice axis: {axis!r}")
    if not np.isfinite(eye_distance) or eye_distance <= 0:
        raise CameraValidationError("eye distance must be finite and positive")

    lattice = np.asarray(cell, dtype=float)
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise CameraValidationError("cell must be a finite 3x3 array")

    index = axis_indices[axis]
    direction = lattice[index].copy()
    if np.linalg.norm(direction) <= 1e-12:
        direction = np.eye(3)[index]
    direction /= np.linalg.norm(direction)

    # a/b 视角尽量以 c 轴向上，c 视角以 b 轴向上。
    preferred_indices = (2, 1, 0) if axis != "c" else (1, 0, 2)
    candidates = [lattice[item] for item in preferred_indices]
    candidates.extend(np.eye(3)[item] for item in preferred_indices)
    up = None
    for candidate in candidates:
        projected = candidate - np.dot(candidate, direction) * direction
        length = np.linalg.norm(projected)
        if length > 1e-12:
            up = projected / length
            break
    if up is None:
        raise CameraValidationError("cannot construct camera up from cell")

    return CameraState(
        eye=tuple(float(value) for value in direction * eye_distance),
        up=tuple(float(value) for value in up),
        center=(0.0, 0.0, 0.0),
    )


@dataclass(frozen=True)
class ApplyCameraEvent:
    """浏览器点击“应用当前视角”后提交的规范化事件。"""

    event_id: str
    structure_id: str
    camera: CameraState


@dataclass(frozen=True)
class AtomSelectionEvent:
    """浏览器点击原子后提交的规范化选择事件。"""

    event_id: str
    structure_id: str
    atom_index: int
    atom_symbol: str


@dataclass(frozen=True)
class AtomSelectionBatchEvent:
    """浏览器确认前端临时选区后提交的完整批量选择。"""

    event_id: str
    structure_id: str
    atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class AcceptedCameraEvent:
    """通过身份校验且已转换矩阵的相机事件。"""

    event_id: str
    camera: CameraState
    rotation_matrix: np.ndarray


@dataclass(frozen=True)
class AppliedViewState:
    """Python 侧唯一获准驱动 3D 与 2D 的完整视角。"""

    camera: CameraState
    rotation_matrix: np.ndarray
    event_id: str | None
    view_revision: str


def initial_applied_view(
    rotation: str,
    *,
    view_revision: str,
) -> AppliedViewState:
    """从视角预设建立一致的 Plotly 相机和 2D 旋转矩阵。"""
    matrix = np.asarray(rotate(rotation), dtype=float)
    return AppliedViewState(
        camera=rotation_matrix_to_camera(matrix),
        rotation_matrix=matrix,
        event_id=None,
        view_revision=view_revision,
    )


def update_applied_view(
    current: AppliedViewState,
    accepted: AcceptedCameraEvent,
) -> AppliedViewState:
    """用一次已验证事件原子化替换已应用视角。"""
    return dataclasses.replace(
        current,
        camera=accepted.camera,
        rotation_matrix=accepted.rotation_matrix.copy(),
        event_id=accepted.event_id,
        view_revision=f"camera:{accepted.event_id}",
    )


def store_applied_view_state(
    session_state: Any,
    applied: AppliedViewState,
) -> None:
    """集中写入 Streamlit session state 的已应用视角字段。"""
    session_state["meia_applied_camera"] = applied.camera
    session_state["meia_applied_rotation_matrix"] = applied.rotation_matrix.copy()
    session_state["meia_processed_viewer_event_id"] = applied.event_id
    session_state["meia_view_revision"] = applied.view_revision


def load_applied_view_state(session_state: Any) -> AppliedViewState:
    """从固定 session state 键重建完整已应用视角。"""
    return AppliedViewState(
        camera=session_state["meia_applied_camera"],
        rotation_matrix=np.asarray(
            session_state["meia_applied_rotation_matrix"],
            dtype=float,
        ).copy(),
        event_id=session_state["meia_processed_viewer_event_id"],
        view_revision=session_state["meia_view_revision"],
    )


def structure_id_from_bytes(content: bytes) -> str:
    """用上传文件原始内容生成稳定结构身份。"""
    return sha256(content).hexdigest()


def parse_apply_camera_event(value: object) -> ApplyCameraEvent | None:
    """解析自定义组件返回值，不向调用方泄漏原始事件字典。"""
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("event_type") != "apply_camera":
        raise ViewerEventError("unsupported viewer event")

    event_id = value.get("event_id")
    structure_id = value.get("structure_id")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ViewerEventError("viewer event_id must be a non-empty string")
    if not isinstance(structure_id, str) or not structure_id.strip():
        raise ViewerEventError("viewer structure_id must be a non-empty string")

    try:
        raw_camera = value["camera"]
        camera = CameraState.from_mapping(raw_camera)
    except (KeyError, CameraValidationError, TypeError) as exc:
        raise ViewerEventError(f"invalid viewer camera: {exc}") from exc
    return ApplyCameraEvent(
        event_id=event_id,
        structure_id=structure_id,
        camera=camera,
    )


def parse_atom_selection_event(value: object) -> AtomSelectionEvent | None:
    """解析 3D 原子点选事件，并拒绝隐式数值转换和空身份。"""
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("event_type") != "select_atom":
        raise ViewerEventError("unsupported viewer event")

    event_id = value.get("event_id")
    structure_id = value.get("structure_id")
    atom_index = value.get("atom_index")
    atom_symbol = value.get("atom_symbol")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ViewerEventError("viewer event_id must be a non-empty string")
    if not isinstance(structure_id, str) or not structure_id.strip():
        raise ViewerEventError("viewer structure_id must be a non-empty string")
    if isinstance(atom_index, bool) or not isinstance(atom_index, int):
        raise ViewerEventError("viewer atom_index must be an integer")
    if not isinstance(atom_symbol, str) or not atom_symbol.strip():
        raise ViewerEventError("viewer atom_symbol must be a non-empty string")

    return AtomSelectionEvent(
        event_id=event_id,
        structure_id=structure_id,
        atom_index=atom_index,
        atom_symbol=atom_symbol,
    )


def parse_atom_selection_batch_event(
    value: object,
) -> AtomSelectionBatchEvent | None:
    """解析一次前端确认的完整多选集合，不接受隐式索引转换。"""
    if value is None:
        return None
    if not isinstance(value, Mapping) or value.get("event_type") != "select_atoms":
        raise ViewerEventError("unsupported viewer event")

    event_id = value.get("event_id")
    structure_id = value.get("structure_id")
    raw_indices = value.get("atom_indices")
    if not isinstance(event_id, str) or not event_id.strip():
        raise ViewerEventError("viewer event_id must be a non-empty string")
    if not isinstance(structure_id, str) or not structure_id.strip():
        raise ViewerEventError("viewer structure_id must be a non-empty string")
    if not isinstance(raw_indices, list) or any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in raw_indices
    ):
        raise ViewerEventError(
            "viewer atom_indices must contain non-negative integers"
        )
    indices = tuple(raw_indices)
    if indices != tuple(sorted(set(indices))):
        raise ViewerEventError("viewer atom_indices must be canonical")
    return AtomSelectionBatchEvent(
        event_id=event_id,
        structure_id=structure_id,
        atom_indices=indices,
    )


def accept_apply_camera_event(
    value: object,
    *,
    current_structure_id: str,
    processed_event_id: str | None,
) -> AcceptedCameraEvent | None:
    """校验结构与事件 ID，并生成唯一一次的已接受相机更新。

    Streamlit 在组件参数变化后的首轮 rerun 仍可能返回上一构型的值；
    这种可预期的迟到事件应静默丢弃，格式错误仍由解析层报告。
    """
    event = parse_apply_camera_event(value)
    if event is None or event.event_id == processed_event_id:
        return None
    if event.structure_id != current_structure_id:
        return None

    return AcceptedCameraEvent(
        event_id=event.event_id,
        camera=event.camera,
        rotation_matrix=camera_to_rotation_matrix(event.camera),
    )


def accept_atom_selection_event(
    value: object,
    *,
    current_structure_id: str,
    processed_event_id: str | None,
    atom_symbols: Sequence[str],
) -> AtomSelectionEvent | None:
    """接受一次属于当前结构、且与当前原子表一致的点选事件。"""
    event = parse_atom_selection_event(value)
    if event is None or event.event_id == processed_event_id:
        return None
    if event.structure_id != current_structure_id:
        return None
    if not 0 <= event.atom_index < len(atom_symbols):
        raise ViewerEventError("viewer atom_index is out of range")
    if atom_symbols[event.atom_index] != event.atom_symbol:
        raise ViewerEventError("viewer atom_symbol does not match atom_index")
    return event


def accept_atom_selection_batch_event(
    value: object,
    *,
    current_structure_id: str,
    processed_event_id: str | None,
    atom_count: int,
) -> AtomSelectionBatchEvent | None:
    """接受一次属于当前构型且尚未处理的完整多选集合。"""
    event = parse_atom_selection_batch_event(value)
    if event is None or event.event_id == processed_event_id:
        return None
    if event.structure_id != current_structure_id:
        return None
    if any(index >= atom_count for index in event.atom_indices):
        raise ViewerEventError("viewer atom_indices contain an out-of-range index")
    return event
