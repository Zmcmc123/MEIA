"""化学键草稿、已应用配置和选择状态的纯函数测试。"""

from dataclasses import replace

import pytest
from ase import Atoms

from meia.bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondRuleError,
    BondSettings,
)
from meia.bond_state import (
    apply_bond_draft,
    initialize_bond_state,
    load_bond_state,
    reset_bond_state_for_structure,
    set_bond_draft,
    store_bond_state,
)


def settings(*, enabled=True, overrides=()):
    return BondSettings(
        pair_rules=(BondPairRule("C", "O", 0.8, 1.8, enabled=enabled),),
        atom_overrides=overrides,
    )


def test_valid_draft_atomically_replaces_applied_and_increments_revision():
    """应用按钮对应的状态转换应一次替换整份规则。"""
    atoms = Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]])
    initial = initialize_bond_state("structure-a", settings(enabled=True))
    initial = set_bond_draft(initial, settings(enabled=False))

    updated = apply_bond_draft(initial, atoms)

    assert updated.applied == settings(enabled=False)
    assert updated.draft == updated.applied
    assert updated.revision == 1


def test_invalid_draft_does_not_replace_applied_bond_settings():
    """与当前原子表不一致的例外不得破坏上一份已应用配置。"""
    atoms = Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]])
    valid = settings()
    invalid = settings(
        overrides=(AtomBondOverride(1, "C", "C", "O", "show"),),
    )
    state = set_bond_draft(
        initialize_bond_state("structure-a", valid),
        invalid,
    )

    with pytest.raises(BondRuleError):
        apply_bond_draft(state, atoms)

    assert state.applied == valid
    assert state.revision == 0


def test_structure_change_reinitializes_rules_and_selection():
    """新构型必须丢弃旧规则草稿、临时选择及迟到事件身份。"""
    first = initialize_bond_state("structure-a", settings())
    first = replace(
        first,
        selected_atom_index=1,
        processed_selection_event_id="s1",
    )
    replacement = BondSettings(
        pair_rules=(BondPairRule("H", "O", 0.5, 1.2),),
    )

    second = reset_bond_state_for_structure(first, "structure-b", replacement)

    assert second.structure_id == "structure-b"
    assert second.applied == replacement
    assert second.draft == replacement
    assert second.selected_atom_index is None
    assert second.processed_selection_event_id is None
    assert second.revision == 0


def test_session_adapter_uses_only_fixed_bond_state_keys():
    """Streamlit 重跑边界应集中在稳定键名，避免部分状态被漏写。"""
    session = {}
    expected = replace(
        initialize_bond_state("structure-a", settings()),
        selected_atom_index=1,
        processed_selection_event_id="s1",
    )

    store_bond_state(session, expected)
    actual = load_bond_state(session)

    assert actual == expected
    assert set(session) == {
        "meia_bond_structure_id",
        "meia_applied_bond_settings",
        "meia_draft_bond_settings",
        "meia_bond_revision",
        "meia_selected_atom_index",
        "meia_processed_selection_event_id",
    }
