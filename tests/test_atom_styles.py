"""具体原子绝对色彩强度与草稿状态测试。"""

import importlib

import numpy as np
import pytest
from ase import Atoms
from matplotlib.colors import to_hex
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse

from meia.config import RenderConfig
from meia.bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondSettings,
    OverrideVisibility,
    resolve_bonds,
)
from meia.geometry import compute_bond_geometries
from meia.i18n import I18n, Locale
from meia.periodic_display import CellPeriodicSettings, build_periodic_display
from meia.projection import project_periodic_display
from meia.renderer import render


def _styles_module():
    """在测试体内导入，使缺失的新模块表现为需求失败而非收集错误。"""
    return importlib.import_module("meia.atom_styles")


def test_selection_expression_and_index_have_exact_english_diagnostics():
    styles = _styles_module()
    with pytest.raises(ValueError) as token_error:
        styles.parse_atom_index_expression("1, bad", 3)
    assert I18n(Locale.EN).error_text(
        token_error.value, "selection.apply_failed"
    ) == "Invalid atom-number token: 'bad'."

    settings = styles.AtomSelectionSettings(selected_atom_indices=(3,))
    with pytest.raises(ValueError) as index_error:
        styles.validate_atom_selection_settings(Atoms("HHO"), settings)
    assert I18n(Locale.EN).error_text(
        index_error.value, "selection.apply_failed"
    ) == "Atom number 4 is outside the current structure of 3 atoms."


def test_color_strength_blends_immutable_base_toward_white():
    """色彩强度必须从原色计算，并覆盖灰白填充和深色描边。"""
    styles = _styles_module()

    assert styles.apply_color_strength("#DFA3A3", 0.30) == "#F5E3E3"
    assert styles.apply_color_strength("#E6E6E5", 0.30) == "#F7F7F7"
    assert styles.apply_color_strength("#231815", 0.30) == "#BDBAB9"
    assert styles.apply_color_strength("#DFA3A3", 0.0) == "#FFFFFF"
    assert styles.apply_color_strength("#DFA3A3", 1.0) == "#DFA3A3"


@pytest.mark.parametrize("strength", [-0.01, 1.01, np.nan, True, "0.3"])
def test_color_strength_rejects_values_outside_absolute_unit_interval(strength):
    styles = _styles_module()

    with pytest.raises(ValueError, match="色彩强度"):
        styles.apply_color_strength("#DFA3A3", strength)


def test_strength_operation_compacts_against_profile_default():
    styles = _styles_module()
    atoms = Atoms("HH", positions=np.zeros((2, 3)))
    current = styles.AtomSelectionSettings(
        selected_atom_indices=(0,),
        color_strengths=(styles.AtomColorStrength(0, "H", 1.0),),
        default_color_strength=0.30,
    )

    restored_to_default = styles.apply_atom_selection_operation(
        atoms,
        current,
        styles.AtomSelectionOperation(strength=0.30),
        available_pairs=(),
    )
    assert restored_to_default.color_strengths == ()

    retained_full_strength = styles.apply_atom_selection_operation(
        atoms,
        restored_to_default,
        styles.AtomSelectionOperation(strength=1.0),
        available_pairs=(),
    )
    assert [
        (item.atom_index, item.strength)
        for item in retained_full_strength.color_strengths
    ] == [(0, 1.0)]


def test_render_config_fills_from_default_strength_then_applies_exceptions():
    config = RenderConfig(
        atom_default_color_strength=0.30,
        atom_color_strengths={0: 1.0},
    )

    assert np.allclose(config.get_atom_color_strengths(3), [1.0, 0.30, 0.30])
    assert config.get_atom_outline_colors(3) == [
        "#231815",
        "#BDBAB9",
        "#BDBAB9",
    ]


def test_render_config_applies_strength_once_to_fill_and_atom_outline():
    """重复读取颜色不得在已经弱化的结果上再次衰减。"""
    config = RenderConfig(
        custom_colors={"O": "#DFA3A3", "H": "#E6E6E5"},
        atom_color_strengths={0: 0.30},
    )

    first = config.get_atom_colors(["O", "H"])
    second = config.get_atom_colors(["O", "H"])

    assert first == ["#F5E3E3", "#E6E6E5"]
    assert second == first
    assert config.get_atom_outline_colors(2) == ["#BDBAB9", "#231815"]
    assert np.allclose(config.get_atom_color_strengths(2), [0.30, 1.0])


def test_render_config_preserves_legacy_positional_argument_order():
    """新增原子样式字段必须追加，不能静默改变旧位置参数的含义。"""
    allowed_pairs = {("H", "O")}
    config = RenderConfig(
        0.7,
        0.4,
        1.1,
        0.35,
        0.25,
        False,
        0.45,
        "#111111",
        "#222222",
        0.2,
        False,
        300,
        "45x",
        None,
        1,
        {"H": "#EEEEEE"},
        1.25,
        640,
        allowed_pairs,
    )

    assert config.scale == 1.25
    assert config.maxwidth == 640
    assert config.allowed_pairs == allowed_pairs
    assert config.atom_color_strengths == {}
    assert config.atom_color_overrides == {}


def test_specific_atom_color_precedes_element_color_then_strength():
    """具体原子色必须先于元素色，且绝对强度只在最终基础色上应用一次。"""
    config = RenderConfig(
        custom_colors={"O": "#E5A6A6"},
        atom_color_strengths={0: 0.5},
        atom_color_overrides={0: "#336699"},
    )

    assert config.get_atom_colors(["O", "O"]) == ["#99B2CC", "#E5A6A6"]
    assert config.get_atom_outline_colors(2) == ["#918C8A", "#231815"]


def test_atom_color_state_overwrites_absolute_targets_and_applies_atomically():
    """对同一批原子重复设置 30% 必须保持精确 30%，不能变为 9%。"""
    styles = _styles_module()
    atoms = Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]])
    state = styles.initialize_atom_color_state("structure-a")
    state = styles.set_selected_atom_indices(state, (0, 1), len(atoms))

    once = styles.set_selected_color_strength(state, atoms, 0.30)
    twice = styles.set_selected_color_strength(once, atoms, 0.30)

    assert twice.draft == once.draft
    assert styles.color_strength_mapping(twice.draft) == {0: 0.30, 1: 0.30}
    assert twice.is_dirty is True

    applied = styles.apply_atom_color_draft(twice, atoms)
    assert applied.applied == twice.draft
    assert applied.is_dirty is False
    assert applied.revision == 1


def test_atom_color_state_validates_symbol_identity_and_restores_full_strength():
    styles = _styles_module()
    atoms = Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]])
    invalid = styles.AtomColorStrength(0, "O", 0.30)

    with pytest.raises(ValueError, match="元素"):
        styles.validate_atom_color_strengths(atoms, (invalid,))

    state = styles.initialize_atom_color_state("structure-a")
    state = styles.set_selected_atom_indices(state, (0,), len(atoms))
    state = styles.set_selected_color_strength(state, atoms, 0.30)
    restored = styles.set_selected_color_strength(state, atoms, 1.0)

    assert restored.draft == ()


def test_batch_selection_supports_toggle_ranges_invert_and_session_round_trip():
    styles = _styles_module()
    atoms = Atoms("HHOCO", positions=np.zeros((5, 3)))
    state = styles.initialize_atom_color_state("structure-a")

    assert styles.parse_atom_index_expression("1-3, 5, 3", len(atoms)) == (
        0,
        1,
        2,
        4,
    )
    selected = styles.set_selected_atom_indices(state, (0, 2), len(atoms))
    toggled = styles.toggle_selected_atom(selected, 1, len(atoms))
    inverted = styles.invert_selected_atoms(toggled, len(atoms))
    assert inverted.selected_atom_indices == (3, 4)

    session = {}
    styles.store_atom_color_state(session, inverted)
    assert styles.load_atom_color_state(session) == inverted

    with pytest.raises(ValueError, match="原子序号"):
        styles.parse_atom_index_expression("0, 2", len(atoms))
    with pytest.raises(ValueError, match="原子序号"):
        styles.parse_atom_index_expression("4-2", len(atoms))


def test_mixed_selection_applies_only_compatible_bond_overrides():
    """混合元素选区中，每条元素对例外只应写入该元素对中的原子。"""
    styles = _styles_module()
    atoms = Atoms("HOCa", positions=np.zeros((3, 3)))
    state = styles.AtomSelectionSettings(selected_atom_indices=(0, 1, 2))
    operation = styles.AtomSelectionOperation(
        color_action="set",
        color="#6699CC",
        strength=0.30,
        bond_visibility={
            ("H", "O"): OverrideVisibility.HIDE,
            ("Ca", "O"): OverrideVisibility.SHOW,
        },
    )

    updated = styles.apply_atom_selection_operation(
        atoms,
        state,
        operation,
        available_pairs=(("H", "O"), ("Ca", "O")),
    )

    assert [item.atom_index for item in updated.color_overrides] == [0, 1, 2]
    assert styles.atom_color_override_mapping(updated.color_overrides) == {
        0: "#6699CC",
        1: "#6699CC",
        2: "#6699CC",
    }
    assert styles.color_strength_mapping(updated.color_strengths) == {
        0: 0.30,
        1: 0.30,
        2: 0.30,
    }
    assert {(item.atom_index, item.pair) for item in updated.bond_overrides} == {
        (0, ("H", "O")),
        (1, ("H", "O")),
        (1, ("Ca", "O")),
        (2, ("Ca", "O")),
    }


def test_selected_atoms_can_be_hidden_restored_and_given_hydrogen_bond_rules():
    styles = _styles_module()
    atoms = Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]])
    selected = styles.AtomSelectionSettings(selected_atom_indices=(0, 1))
    hidden = styles.apply_atom_selection_operation(
        atoms,
        selected,
        styles.AtomSelectionOperation(
            visibility_action="hide",
            hydrogen_bond_visibility=OverrideVisibility.HIDE,
        ),
        available_pairs=(("H", "O"),),
    )
    assert {(item.atom_index, item.atom_symbol) for item in hidden.hidden_atoms} == {
        (0, "H"),
        (1, "O"),
    }
    assert {item.atom_index for item in hidden.hydrogen_bond_overrides} == {0, 1}
    restored = styles.apply_atom_selection_operation(
        atoms,
        hidden,
        styles.AtomSelectionOperation(
            visibility_action="show",
            hydrogen_bond_visibility=OverrideVisibility.INHERIT,
        ),
        available_pairs=(("H", "O"),),
    )
    assert restored.hidden_atoms == ()
    assert restored.hydrogen_bond_overrides == ()


def test_atom_selection_inherit_and_full_strength_remove_only_selected_overrides():
    styles = _styles_module()
    atoms = Atoms("HO", positions=np.zeros((2, 3)))
    state = styles.AtomSelectionSettings(
        selected_atom_indices=(0,),
        color_overrides=(
            styles.AtomColorOverride(0, "H", "#112233"),
            styles.AtomColorOverride(1, "O", "#445566"),
        ),
        color_strengths=(
            styles.AtomColorStrength(0, "H", 0.2),
            styles.AtomColorStrength(1, "O", 0.4),
        ),
        bond_overrides=(
            AtomBondOverride(0, "H", "H", "O", OverrideVisibility.HIDE),
            AtomBondOverride(1, "O", "H", "O", OverrideVisibility.SHOW),
        ),
    )
    operation = styles.AtomSelectionOperation(
        color_action="inherit",
        strength=1.0,
        bond_visibility={("H", "O"): OverrideVisibility.INHERIT},
    )

    updated = styles.apply_atom_selection_operation(
        atoms,
        state,
        operation,
        available_pairs=(("H", "O"),),
    )

    assert [(item.atom_index, item.color) for item in updated.color_overrides] == [
        (1, "#445566")
    ]
    assert [(item.atom_index, item.strength) for item in updated.color_strengths] == [
        (1, 0.4)
    ]
    assert [(item.atom_index, item.visibility) for item in updated.bond_overrides] == [
        (1, OverrideVisibility.SHOW)
    ]


def test_atom_selection_operation_rejects_invalid_color_without_partial_result():
    styles = _styles_module()
    atoms = Atoms("HO", positions=np.zeros((2, 3)))
    original = styles.AtomSelectionSettings(selected_atom_indices=(0, 1))

    with pytest.raises(ValueError, match="颜色"):
        operation = styles.AtomSelectionOperation(
            color_action="set",
            color="not-a-color",
            strength=0.3,
        )
        styles.apply_atom_selection_operation(
            atoms,
            original,
            operation,
            available_pairs=(("H", "O"),),
        )

    assert original.color_overrides == ()
    assert original.color_strengths == ()


def test_atom_selection_operation_rejects_stale_atom_identity():
    styles = _styles_module()
    atoms = Atoms("HO", positions=np.zeros((2, 3)))
    stale = styles.AtomSelectionSettings(
        selected_atom_indices=(0,),
        color_overrides=(styles.AtomColorOverride(0, "O", "#112233"),),
    )

    with pytest.raises(ValueError, match="元素"):
        styles.apply_atom_selection_operation(
            atoms,
            stale,
            styles.AtomSelectionOperation(),
            available_pairs=(("H", "O"),),
        )


def test_replace_selected_indices_uses_one_type_for_single_and_multiple_selection():
    styles = _styles_module()
    current = styles.AtomSelectionSettings(
        color_strengths=(styles.AtomColorStrength(0, "H", 0.3),),
    )

    single = styles.replace_selected_indices(current, (1,), atom_count=3)
    multiple = styles.replace_selected_indices(current, (0, 2), atom_count=3)

    assert isinstance(single, styles.AtomSelectionSettings)
    assert isinstance(multiple, styles.AtomSelectionSettings)
    assert single.selected_atom_indices == (1,)
    assert multiple.selected_atom_indices == (0, 2)
    assert single.color_strengths == current.color_strengths
    assert multiple.color_strengths == current.color_strengths


def test_2d_strength_updates_atom_outline_half_bond_and_whole_bond_outline():
    """整键描边必须取两端较低强度，同时保留六对象键结构。"""
    atoms = Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]])
    config = RenderConfig(
        show_unit_cell=0,
        custom_colors={"H": "#E6E6E5", "O": "#DFA3A3"},
        atom_color_strengths={0: 0.30},
    )
    settings = BondSettings(
        pair_rules=(BondPairRule("H", "O", 0.0, 1.2),),
    )
    resolution = resolve_bonds(atoms, settings)
    display = build_periodic_display(
        atoms,
        resolution.matched,
        CellPeriodicSettings(show_unit_cell=0),
    )
    projection = project_periodic_display(atoms, display, config, frozenset())

    geometries = compute_bond_geometries(
        display.bond_instances,
        projection,
        config,
    )
    figure = render(projection, geometries, config)

    assert projection.colors == ["#F7F7F7", "#DFA3A3"]
    assert projection.outline_colors == ["#BDBAB9", "#231815"]
    assert np.allclose(projection.color_strengths, [0.30, 1.0])
    assert len(geometries) == 1
    assert geometries[0].stroke_color == "#BDBAB9"

    atom_circles = [patch for patch in figure.axes[0].patches if type(patch) is Circle]
    bond_caps = [patch for patch in figure.axes[0].patches if type(patch) is Ellipse]
    bond_lines = [artist for artist in figure.axes[0].lines if isinstance(artist, Line2D)]
    assert [to_hex(circle.get_edgecolor()).upper() for circle in atom_circles] == [
        "#BDBAB9",
        "#231815",
    ]
    assert len(bond_caps) == 2
    assert {to_hex(cap.get_edgecolor()).upper() for cap in bond_caps} == {"#BDBAB9"}
    assert len(bond_lines) == 2
    assert {to_hex(line.get_color()).upper() for line in bond_lines} == {"#BDBAB9"}

    import matplotlib.pyplot as plt

    plt.close(figure)
