"""元素对规则、具体原子例外与周期距离解析测试。"""

import numpy as np
import pytest
from ase import Atoms

from meia.bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondOverrideConflict,
    BondRuleError,
    BondSettings,
    BondStyle,
    OverrideVisibility,
    count_drawable_bonds,
    find_override_conflicts,
    initialize_bond_settings,
    normalize_element_pair,
    reapply_bond_visibility,
    resolve_bonds,
)
from meia.config import RenderConfig
from meia.i18n import I18n, Locale


def test_invalid_bond_range_and_stroke_have_exact_english_diagnostics():
    with pytest.raises(BondRuleError) as range_error:
        BondPairRule("H", "O", 1.2, 0.8)
    assert I18n(Locale.EN).error_text(
        range_error.value, "bonds.apply_failed"
    ) == (
        "The maximum bond distance (0.8 Å) must be at least the minimum "
        "(1.2 Å)."
    )

    with pytest.raises(BondRuleError) as stroke_error:
        BondStyle(stroke_width=-0.25)
    assert I18n(Locale.EN).error_text(
        stroke_error.value, "bonds.apply_failed"
    ) == "The bond outline width must be 0 or greater; received -0.25."


def test_element_pair_is_canonical_and_validated():
    assert normalize_element_pair("O", "Ca") == ("Ca", "O")
    assert BondPairRule("O", "Ca", 2.1, 2.8).pair == ("Ca", "O")

    with pytest.raises(BondRuleError, match="元素"):
        BondPairRule("O", "NotAnElement", 0.0, 2.0)


def test_rule_and_override_store_canonical_element_order():
    """规范化不仅用于比较，也必须成为可序列化对象的稳定值。"""
    rule = BondPairRule("O", "Ca", 2.1, 2.8)
    override = AtomBondOverride(0, "O", "O", "Ca", "show")

    assert (rule.element_a, rule.element_b) == ("Ca", "O")
    assert (override.element_a, override.element_b) == ("Ca", "O")


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (-0.1, 2.0, "最小距离"),
        (3.0, 2.0, "最大距离"),
        (0.0, np.inf, "有限"),
        (False, 2.0, "数值"),
    ],
)
def test_pair_rule_validates_closed_distance_range(minimum, maximum, message):
    with pytest.raises(BondRuleError, match=message):
        BondPairRule("Ca", "O", minimum, maximum)


def test_pair_rule_validates_periodic_participation_boolean():
    with pytest.raises(BondRuleError, match="周期整理开关"):
        BondPairRule(
            "Ca", "O", 0.0, 2.8,
            participates_in_periodic_unwrap="yes",
        )


def test_duplicate_normalized_pairs_are_rejected():
    with pytest.raises(BondRuleError, match="重复"):
        BondSettings(
            pair_rules=(
                BondPairRule("Ca", "O", 0.0, 2.8),
                BondPairRule("O", "Ca", 0.0, 3.0),
            )
        )


def test_hidden_pair_can_be_shown_for_one_atom_but_not_outside_range():
    atoms = Atoms(
        symbols=["Ca", "O", "Ca"],
        positions=[[0.0, 0.0, 0.0], [2.3, 0.0, 0.0], [7.0, 0.0, 0.0]],
    )
    settings = BondSettings(
        pair_rules=(BondPairRule("Ca", "O", 2.1, 2.8, enabled=False),),
        atom_overrides=(
            AtomBondOverride(
                atom_index=1,
                atom_symbol="O",
                element_a="Ca",
                element_b="O",
                visibility=OverrideVisibility.SHOW,
            ),
        ),
    )

    resolution = resolve_bonds(atoms, settings)

    assert [(bond.i, bond.j) for bond in resolution.matched] == [(0, 1)]
    assert [(bond.i, bond.j) for bond in resolution.visible] == [(0, 1)]
    assert resolution.visible[0].visibility_source == "atom_show"


def test_reapplying_visibility_preserves_matched_bond_identity_and_counts():
    atoms = Atoms("CaO", positions=[[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]])
    initially_visible = BondSettings(
        pair_rules=(BondPairRule("Ca", "O", 2.1, 2.8, enabled=True),),
    )
    resolution = resolve_bonds(atoms, initially_visible)

    reapplied = reapply_bond_visibility(
        resolution,
        BondSettings(
            pair_rules=(BondPairRule("Ca", "O", 2.1, 2.8, enabled=False),),
        ),
    )

    original = resolution.matched[0]
    updated = reapplied.matched[0]
    assert (
        updated.i,
        updated.j,
        updated.offset,
        updated.distance,
        updated.pair,
        updated.bond_id,
    ) == (
        original.i,
        original.j,
        original.offset,
        original.distance,
        original.pair,
        original.bond_id,
    )
    assert reapplied.match_counts is resolution.match_counts
    assert original.visible is True
    assert original.visibility_source == "pair_enabled"
    assert updated.visible is False
    assert updated.visibility_source == "pair_disabled"
    assert reapplied.visible == ()


def test_force_hide_wins_when_both_endpoints_conflict():
    atoms = Atoms("CaO", positions=[[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]])
    settings = BondSettings(
        pair_rules=(BondPairRule("Ca", "O", 2.1, 2.8, enabled=True),),
        atom_overrides=(
            AtomBondOverride(
                0, "Ca", "Ca", "O", OverrideVisibility.HIDE
            ),
            AtomBondOverride(
                1, "O", "Ca", "O", OverrideVisibility.SHOW
            ),
        ),
    )

    resolution = resolve_bonds(atoms, settings)

    assert len(resolution.matched) == 1
    assert resolution.visible == ()
    assert resolution.matched[0].visibility_source == "atom_hide"


def test_override_cannot_bypass_pair_distance_range():
    atoms = Atoms("CaO", positions=[[0.0, 0.0, 0.0], [3.1, 0.0, 0.0]])
    settings = BondSettings(
        pair_rules=(BondPairRule("Ca", "O", 2.1, 2.8, enabled=False),),
        atom_overrides=(
            AtomBondOverride(1, "O", "Ca", "O", OverrideVisibility.SHOW),
        ),
    )

    assert resolve_bonds(atoms, settings).matched == ()


def test_distance_limits_are_inclusive():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    settings = BondSettings(
        pair_rules=(BondPairRule("C", "O", 1.2, 1.2, enabled=True),),
    )

    resolution = resolve_bonds(atoms, settings)

    assert len(resolution.visible) == 1
    assert resolution.visible[0].distance == pytest.approx(1.2)


def test_periodic_resolution_preserves_minimum_image_offset():
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.6, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    settings = BondSettings(
        pair_rules=(BondPairRule("H", "H", 0.4, 0.6, enabled=True),),
    )

    resolution = resolve_bonds(atoms, settings)

    assert len(resolution.visible) == 1
    assert resolution.visible[0].distance == pytest.approx(0.5)
    assert resolution.visible[0].offset == (-1, 0, 0)


def test_single_atom_can_bond_to_its_periodic_image_once():
    """非零晶胞偏移的 i==j 候选是周期键，不是自相互作用噪声。"""
    atoms = Atoms(
        "H",
        positions=[[0.0, 0.0, 0.0]],
        cell=[1.0, 8.0, 8.0],
        pbc=[True, False, False],
    )
    settings = BondSettings(
        pair_rules=(BondPairRule("H", "H", 0.9, 1.1),),
    )

    resolution = resolve_bonds(atoms, settings)

    assert len(resolution.visible) == 1
    assert resolution.visible[0].i == resolution.visible[0].j == 0
    assert resolution.visible[0].offset == (-1, 0, 0)
    assert resolution.visible[0].distance == pytest.approx(1.0)


def test_atom_override_symbol_must_match_structure():
    atoms = Atoms("CaO", positions=[[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]])
    settings = BondSettings(
        pair_rules=(BondPairRule("Ca", "O", 2.1, 2.8),),
        atom_overrides=(
            AtomBondOverride(1, "Ca", "Ca", "O", OverrideVisibility.HIDE),
        ),
    )

    with pytest.raises(BondRuleError, match="元素不一致"):
        resolve_bonds(atoms, settings)


def test_default_initialization_keeps_only_currently_matched_pair_types():
    atoms = Atoms(
        symbols=["C", "O", "Ca"],
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [8.0, 0.0, 0.0]],
    )

    settings = initialize_bond_settings(atoms, RenderConfig(bond_cutoff=1.0))

    assert [rule.pair for rule in settings.pair_rules] == [("C", "O")]
    assert settings.pair_rules[0].min_distance == 0.0
    assert settings.pair_rules[0].enabled is True
    assert settings.style.width_ratio == 0.45
    assert settings.style.stroke_width == 0.25


def test_default_initialization_matches_normal_oh_distance():
    """默认 H–O 阈值必须覆盖常见但略大于 ASE 半径和的 O–H 键。"""
    atoms = Atoms("HO", positions=[[0.0, 0.0, 0.0], [1.08, 0.0, 0.0]])

    settings = initialize_bond_settings(atoms, RenderConfig())
    resolution = resolve_bonds(atoms, settings)

    assert [(rule.pair, rule.max_distance) for rule in settings.pair_rules] == [
        (("H", "O"), pytest.approx(1.164)),
    ]
    assert [(bond.i, bond.j) for bond in resolution.visible] == [(0, 1)]


def test_drawable_counts_distinguish_matched_bonds_hidden_inside_atom_spheres():
    """距离规则命中但显示球相交时，应报告匹配而不是虚构可见键体。"""
    atoms = Atoms("OO", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])
    settings = BondSettings(
        pair_rules=(BondPairRule("O", "O", 0.5, 0.8, enabled=True),),
    )

    resolution = resolve_bonds(atoms, settings)
    drawable = count_drawable_bonds(atoms, settings, radius_scale=0.6)

    assert resolution.match_counts[("O", "O")] == 1
    assert drawable[("O", "O")] == 0


def test_conflicting_endpoint_overrides_are_reported_with_hide_precedence():
    """界面需要可解释地展示同一根键两端相反的强制例外。"""
    atoms = Atoms("CaO", positions=[[0, 0, 0], [2.3, 0, 0]])
    settings = BondSettings(
        pair_rules=(BondPairRule("Ca", "O", 2.1, 2.8),),
        atom_overrides=(
            AtomBondOverride(0, "Ca", "Ca", "O", "hide"),
            AtomBondOverride(1, "O", "Ca", "O", "show"),
        ),
    )

    conflicts = find_override_conflicts(atoms, settings)

    assert conflicts == (
        BondOverrideConflict(
            bond_id="bond_0001_Ca1_O2",
            atom_i=0,
            atom_j=1,
            element_pair=("Ca", "O"),
            hidden_atom_index=0,
            shown_atom_index=1,
        ),
    )
