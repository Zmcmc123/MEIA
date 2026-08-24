"""Streamlit 会话中的活动结构，以及两类 v7 JSON 的显式转换。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from hashlib import sha256
import json

from ase import Atoms

from .i18n import LocalizedError

from .presets import (
    PresetError,
    PresetKind,
    PresetMetadata,
    SCHEMA_VERSION,
    SnapshotStructure,
    StylePreset,
    WorkspaceSnapshot,
)
from .periodic_display import normalize_periodic_settings
from .visual_state import VisualizationState


class WorkspaceError(LocalizedError):
    """工作区或快照确认边界无效。"""


def canonical_structure_bytes(atoms: Atoms) -> bytes:
    """用坐标、晶胞、PBC 和有序元素生成稳定的内存结构表示。"""
    if not isinstance(atoms, Atoms):
        raise TypeError("活动结构必须是 ASE Atoms")
    value = {
        "symbols": atoms.get_chemical_symbols(),
        "positions_angstrom": [
            [float(component) for component in row] for row in atoms.positions
        ],
        "cell_angstrom": [
            [float(component) for component in row] for row in atoms.cell.array
        ],
        "pbc": [bool(value) for value in atoms.pbc],
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def structure_identity(atoms: Atoms) -> str:
    """返回与磁盘来源无关、随结构字段变化的 SHA-256 身份。"""
    return sha256(canonical_structure_bytes(atoms)).hexdigest()


@dataclass(frozen=True)
class ActiveWorkspace:
    """当前页面唯一有权用于渲染和导出的内存结构。"""

    atoms: Atoms
    source_name: str
    source_content: bytes
    origin: str
    structure_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.atoms, Atoms):
            raise TypeError("活动结构必须是 ASE Atoms")
        if not isinstance(self.source_name, str) or not self.source_name.strip():
            raise ValueError("结构来源名称不能为空")
        if not isinstance(self.source_content, bytes):
            raise TypeError("结构来源内容必须是 bytes")
        if self.origin not in {"upload", "snapshot"}:
            raise ValueError("活动结构来源只能是 upload 或 snapshot")
        if self.structure_id != structure_identity(self.atoms):
            raise ValueError("活动结构身份与坐标内容不一致")
        object.__setattr__(self, "atoms", self.atoms.copy())
        object.__setattr__(self, "source_name", self.source_name.strip())

    @classmethod
    def from_upload(
        cls,
        atoms: Atoms,
        source_name: str,
        source_content: bytes,
    ) -> "ActiveWorkspace":
        return cls(
            atoms=atoms,
            source_name=source_name,
            source_content=source_content,
            origin="upload",
            structure_id=structure_identity(atoms),
        )

    @classmethod
    def from_snapshot(cls, snapshot: WorkspaceSnapshot) -> "ActiveWorkspace":
        if not isinstance(snapshot, WorkspaceSnapshot):
            raise TypeError("只能从 WorkspaceSnapshot 激活工作区")
        atoms = snapshot.structure.to_atoms()
        return cls(
            atoms=atoms,
            source_name=snapshot.structure.source_name,
            source_content=canonical_structure_bytes(atoms),
            origin="snapshot",
            structure_id=structure_identity(atoms),
        )


def activate_upload(
    current: ActiveWorkspace | None,
    last_seen_upload_sha256: str | None,
    payload: bytes,
    source_name: str,
    atoms: Atoms,
) -> tuple[ActiveWorkspace, str, bool]:
    """只在上传内容变化时替换活动结构，避免旧 uploader 覆盖快照。"""
    if not isinstance(payload, bytes):
        raise TypeError("上传内容必须是 bytes")
    upload_sha256 = sha256(payload).hexdigest()
    if current is not None and upload_sha256 == last_seen_upload_sha256:
        return current, upload_sha256, False
    return (
        ActiveWorkspace.from_upload(atoms, source_name, payload),
        upload_sha256,
        True,
    )


@dataclass(frozen=True)
class PendingSnapshot:
    """已严格解析、但尚未得到用户覆盖确认的工作状态快照。"""

    snapshot: WorkspaceSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, WorkspaceSnapshot):
            raise TypeError("待确认内容必须是 WorkspaceSnapshot")

    @property
    def source_name(self) -> str:
        return self.snapshot.structure.source_name

    @property
    def atom_count(self) -> int:
        return len(self.snapshot.structure.symbols)


def stage_snapshot(snapshot: WorkspaceSnapshot) -> PendingSnapshot:
    return PendingSnapshot(snapshot)


def activate_snapshot(snapshot: WorkspaceSnapshot) -> ActiveWorkspace:
    return ActiveWorkspace.from_snapshot(snapshot)


def confirm_pending_snapshot(
    pending: PendingSnapshot,
    *,
    confirmed: bool,
) -> tuple[ActiveWorkspace, VisualizationState]:
    if not isinstance(pending, PendingSnapshot):
        raise TypeError("待确认工作状态快照无效")
    if confirmed is not True:
        raise WorkspaceError(
            "必须明确确认后才能覆盖当前结构",
            message_key="workspace.confirmation_required",
        )
    snapshot = pending.snapshot
    atoms = snapshot.structure.to_atoms()
    try:
        normalized_periodic = normalize_periodic_settings(
            atoms,
            snapshot.state.style.cell_periodic,
        )
    except LocalizedError as exc:
        raise PresetError(
            f"工作状态快照无法应用：{exc}",
            message_key=exc.message_key or "workspace.apply_failed",
            message_params=exc.message_params,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise PresetError(
            f"工作状态快照无法应用：{exc}",
            message_key="workspace.apply_failed",
        ) from exc
    normalized_state = replace(
        snapshot.state,
        style=replace(
            snapshot.state.style,
            cell_periodic=normalized_periodic,
        ),
    )
    return activate_snapshot(snapshot), normalized_state


def _metadata(kind: PresetKind, name: str, meia_version: str) -> PresetMetadata:
    return PresetMetadata(
        schema_version=SCHEMA_VERSION,
        preset_kind=kind,
        name=name,
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        meia_version=meia_version,
    )


def build_style_preset(
    state: VisualizationState,
    name: str,
    meia_version: str,
) -> StylePreset:
    if not isinstance(state, VisualizationState):
        raise TypeError("通用风格必须来自 VisualizationState")
    return StylePreset(
        _metadata(PresetKind.STYLE, name, meia_version),
        state.style,
    )


def build_workspace_snapshot(
    atoms: Atoms,
    source_name: str,
    state: VisualizationState,
    name: str,
    meia_version: str,
) -> WorkspaceSnapshot:
    if not isinstance(state, VisualizationState):
        raise TypeError("工作状态快照必须来自 VisualizationState")
    return WorkspaceSnapshot(
        metadata=_metadata(PresetKind.WORKSPACE_SNAPSHOT, name, meia_version),
        structure=SnapshotStructure.from_atoms(atoms, source_name),
        state=state,
    )
