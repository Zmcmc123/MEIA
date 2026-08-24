"""Streamlit 化学键草稿、已应用配置与临时原子选择状态。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ase import Atoms

from .bond_rules import BondSettings, validate_bond_settings


@dataclass(frozen=True)
class AppliedBondState:
    """一份与单个构型绑定的不可变化学键交互状态。"""

    structure_id: str
    applied: BondSettings
    draft: BondSettings
    revision: int = 0
    selected_atom_index: int | None = None
    processed_selection_event_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.structure_id, str) or not self.structure_id.strip():
            raise ValueError("structure_id 必须是非空字符串")
        if not isinstance(self.applied, BondSettings):
            raise TypeError("applied 必须是 BondSettings")
        if not isinstance(self.draft, BondSettings):
            raise TypeError("draft 必须是 BondSettings")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision 必须是整数")
        if self.revision < 0:
            raise ValueError("revision 不能小于 0")
        if self.selected_atom_index is not None:
            if (
                isinstance(self.selected_atom_index, bool)
                or not isinstance(self.selected_atom_index, int)
                or self.selected_atom_index < 0
            ):
                raise ValueError("selected_atom_index 必须是非负整数或 None")
        if self.processed_selection_event_id is not None and (
            not isinstance(self.processed_selection_event_id, str)
            or not self.processed_selection_event_id.strip()
        ):
            raise ValueError("processed_selection_event_id 必须是非空字符串或 None")

    @property
    def is_dirty(self) -> bool:
        """草稿是否尚未应用。"""
        return self.draft != self.applied


def initialize_bond_state(
    structure_id: str,
    settings: BondSettings,
) -> AppliedBondState:
    """为当前构型建立一致的草稿与已应用配置。"""
    return AppliedBondState(
        structure_id=structure_id,
        applied=settings,
        draft=settings,
    )


def reset_bond_state_for_structure(
    current: AppliedBondState,
    structure_id: str,
    settings: BondSettings,
) -> AppliedBondState:
    """切换构型时重建规则，并清空临时选择与旧事件身份。"""
    if not isinstance(current, AppliedBondState):
        raise TypeError("current 必须是 AppliedBondState")
    return initialize_bond_state(structure_id, settings)


def set_bond_draft(
    current: AppliedBondState,
    draft: BondSettings,
) -> AppliedBondState:
    """只替换草稿，不触发渲染配置变化。"""
    if not isinstance(draft, BondSettings):
        raise TypeError("draft 必须是 BondSettings")
    return replace(current, draft=draft)


def apply_bond_draft(
    current: AppliedBondState,
    atoms: Atoms,
) -> AppliedBondState:
    """校验后原子化应用整份草稿；失败时调用方仍持有原状态。"""
    validate_bond_settings(atoms, current.draft)
    return replace(
        current,
        applied=current.draft,
        revision=current.revision + 1,
    )


def store_bond_state(session_state: Any, state: AppliedBondState) -> None:
    """把完整状态集中写入固定的 Streamlit session 键。"""
    session_state["meia_bond_structure_id"] = state.structure_id
    session_state["meia_applied_bond_settings"] = state.applied
    session_state["meia_draft_bond_settings"] = state.draft
    session_state["meia_bond_revision"] = state.revision
    session_state["meia_selected_atom_index"] = state.selected_atom_index
    session_state["meia_processed_selection_event_id"] = (
        state.processed_selection_event_id
    )


def load_bond_state(session_state: Any) -> AppliedBondState:
    """从固定 session 键重建化学键交互状态。"""
    return AppliedBondState(
        structure_id=session_state["meia_bond_structure_id"],
        applied=session_state["meia_applied_bond_settings"],
        draft=session_state["meia_draft_bond_settings"],
        revision=session_state["meia_bond_revision"],
        selected_atom_index=session_state["meia_selected_atom_index"],
        processed_selection_event_id=session_state[
            "meia_processed_selection_event_id"
        ],
    )
