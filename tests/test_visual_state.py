"""已应用可视化状态与统一渲染上下文测试。"""

from dataclasses import replace
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pytest
from ase import Atoms
from matplotlib.patches import Circle

import meia.periodic_display as periodic_display_module
import meia.pipeline as pipeline_module
import meia.view as view_module
import meia.visual_state as visual_state_module

from meia.atom_styles import (
    AtomColorOverride,
    AtomColorStrength,
    AtomHydrogenBondOverride,
    HiddenAtom,
    AtomSelectionSettings,
)
from meia.bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondStrokeStyle,
    BondStyle,
    OverrideVisibility,
)
from meia.projection import project_periodic_display
from meia.view import render_2d
from meia.pipeline import render_atoms
from meia.periodic_display import (
    CellPeriodicSettings,
    PeriodicRange,
    estimate_periodic_atom_instances,
    normalize_periodic_settings,
)
from meia.size_profiles import (
    CovalentSizeProfile,
    RadiusMode,
    SizeProfileSettings,
    UniformSizeProfile,
)
from meia.i18n import I18n, Locale
from meia.view_state import CameraState
from meia.viewer import create_3d_figure as _create_3d_figure
from meia.export import export_figure
from meia.visual_state import (
    AtomCellSettings,
    BondModuleSettings,
    ExportSettings,
    PairRuleDefaults,
    PortableStyle,
    ViewSettings,
    VisualizationState,
    apply_camera_only,
    reset_visual_modules_from_style,
    apply_portable_style,
    merge_pair_rules_for_structure,
    merge_portable_style_for_structure,
    replace_atom_cell,
    resolve_render_context,
)


def create_3d_figure(*args, **kwargs):
    kwargs.setdefault(
        "figure_messages",
        I18n(Locale.ZH_CN).bundle("figure3d"),
    )
    return _create_3d_figure(*args, **kwargs)


def _default_state() -> VisualizationState:
    style = PortableStyle(
        view=ViewSettings(),
        atom_cell=AtomCellSettings(
            element_colors={
                "H": "#E6E6E5",
                "O": "#E5A6A6",
                "Ca": "#9ECC91",
            }
        ),
        bonds=BondModuleSettings(
            pair_rules=(BondPairRule("H", "O", 0.0, 1.2),)
        ),
        export=ExportSettings(),
    )
    return VisualizationState(style=style)

def test_display_radius_changes_do_not_change_bonds_hydrogen_candidates_or_pbc_topology():
    atoms = Atoms(
        "OHO",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [4.8, 0.0, 0.0]],
        cell=[10.0, 8.0, 8.0],
        pbc=[True, False, False],
    )
    state = _default_state_with_ho_bond()
    baseline = resolve_render_context(atoms, state)
    enlarged_state = replace(
        state,
        style=replace(
            state.style,
            size_profiles=SizeProfileSettings(
                active_mode=RadiusMode.UNIFORM,
                uniform=UniformSizeProfile(
                    global_scale=1.2,
                    reference_radius_angstrom=1.5,
                ),
            ),
        ),
    )
    enlarged = resolve_render_context(atoms, enlarged_state)

    assert [bond.bond_id for bond in enlarged.bond_resolution.matched] == [
        bond.bond_id for bond in baseline.bond_resolution.matched
    ]
    assert [bond.bond_id for bond in enlarged.periodic_topology_bonds] == [
        bond.bond_id for bond in baseline.periodic_topology_bonds
    ]
    assert [item.candidate for item in enlarged.hydrogen_bonds] == [
        item.candidate for item in baseline.hydrogen_bonds
    ]
    assert len(enlarged.periodic_display.atom_instances) == len(
        baseline.periodic_display.atom_instances
    )
    for enlarged_instance, baseline_instance in zip(
        enlarged.periodic_display.atom_instances,
        baseline.periodic_display.atom_instances,
    ):
        assert enlarged_instance.source_atom_index == baseline_instance.source_atom_index
        assert enlarged_instance.replica_translation == baseline_instance.replica_translation
        assert enlarged_instance.image_shift == baseline_instance.image_shift
        assert np.allclose(
            enlarged_instance.position,
            baseline_instance.position,
        )
        assert enlarged_instance.instance_id == baseline_instance.instance_id
    assert enlarged.config.get_atom_radii(["O", "H", "O"]) == pytest.approx(
        [1.8, 1.8, 1.8]
    )


def _default_state_with_ho_bond():
    return replace(
        _default_state(),
        style=replace(
            _default_state().style,
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.2),)
            ),
            cell_periodic=CellPeriodicSettings(show_unit_cell=0),
        ),
    )


def _figure_2d_for_context(context):
    return render_2d(
        context.periodic_display.atoms,
        context.config,
        render_context=context,
    )


def _custom_radius_context(atoms):
    state = replace(
        _default_state(),
        style=replace(
            _default_state().style,
            size_profiles=SizeProfileSettings(
                active_mode=RadiusMode.UNIFORM,
                uniform=UniformSizeProfile(
                    global_scale=1.0,
                    reference_radius_angstrom=0.35,
                    reference_overrides_angstrom={"O": 0.6},
                ),
            ),
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.2),)
            ),
            cell_periodic=CellPeriodicSettings(show_unit_cell=0),
        ),
    )
    return resolve_render_context(atoms, state)


def test_custom_display_radii_flow_to_2d_3d_bonds_hydrogen_bonds_and_svg():
    atoms = Atoms(
        "OHO",
        positions=[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[10.0, 8.0, 8.0],
        pbc=[True, False, False],
    )
    context = _custom_radius_context(atoms)
    expected_radii = context.config.get_atom_radii(atoms.get_chemical_symbols())

    projection = project_periodic_display(
        atoms,
        context.periodic_display,
        context.config,
        context.hidden_atom_indices,
    )
    assert np.asarray(projection.radii_2d) / projection.scale == pytest.approx(
        expected_radii[list(projection.source_atom_indices)]
    )

    figure_3d = create_3d_figure(
        atoms,
        context.config,
        render_context=context,
        selected_atom_indices=(1,),
    )
    atom_trace = next(
        trace
        for trace in figure_3d.data
        if trace.meta and trace.meta.get("meia_role") == "atoms"
    )
    assert np.asarray(atom_trace.marker.size) == pytest.approx(
        expected_radii[list(projection.source_atom_indices)] * 15
    )
    highlight_trace = next(
        trace
        for trace in figure_3d.data
        if trace.meta and trace.meta.get("meia_role") == "selection"
    )
    expected_selection_sizes = np.where(
        np.asarray(projection.source_atom_indices) == 1,
        np.asarray(atom_trace.marker.size),
        0.0,
    )
    assert np.asarray(highlight_trace.marker.size) == pytest.approx(
        expected_selection_sizes
    )

    ordinary_traces = [
        trace
        for trace in figure_3d.data
        if trace.meta and trace.meta.get("meia_role") == "bonds"
    ]
    bond = context.periodic_display.bond_instances[0].source_bond
    center_i = atoms.positions[bond.i]
    center_j = atoms.positions[bond.j]
    direction = center_j - center_i
    direction /= np.linalg.norm(direction)
    # ResolvedBond 的 pair 已规范化，i/j 直接给出对应元素索引。
    expected_i = center_i + direction * expected_radii[bond.i]
    expected_j = center_j - direction * expected_radii[bond.j]
    bond_x_points = [
        float(value)
        for trace in ordinary_traces
        for value in trace.x
        if value is not None
    ]
    midpoint_x = float((expected_i[0] + expected_j[0]) / 2)
    assert bond_x_points == pytest.approx(
        [float(expected_i[0]), midpoint_x, midpoint_x, float(expected_j[0])]
    )

    hydrogen_trace = next(
        trace
        for trace in figure_3d.data
        if trace.meta and trace.meta.get("meia_role") == "hydrogen_bonds"
    )
    hydrogen = context.hydrogen_bonds[0].candidate
    center_h = atoms.positions[hydrogen.hydrogen]
    center_acceptor = atoms.positions[hydrogen.acceptor_oxygen]
    direction = center_acceptor - center_h
    direction /= np.linalg.norm(direction)
    expected_h = center_h + direction * expected_radii[hydrogen.hydrogen]
    expected_o = center_acceptor - direction * expected_radii[
        hydrogen.acceptor_oxygen
    ]
    assert list(hydrogen_trace.x)[:2] == pytest.approx(
        [expected_h[0], expected_o[0]]
    )

    figure_2d = render_2d(atoms, context.config, render_context=context)
    svg = export_figure(figure_2d, "svg", context.config).decode("utf-8")
    assert 'data-meia-source-atom-index="0"' in svg
    assert 'data-meia-source-atom-index="1"' in svg
    assert 'data-elements="H-O"' in svg
    assert 'data-hydrogen="1"' in svg



def test_reset_visual_modules_uses_baseline_but_preserves_view_and_export():
    atoms = Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]])
    current = replace(
        _default_state(),
        atom_selection=AtomSelectionSettings(
            selected_atom_indices=(1,),
            hidden_atoms=(HiddenAtom(0, "H"),),
            color_overrides=(AtomColorOverride(1, "O", "#336699"),),
            color_strengths=(AtomColorStrength(1, "O", 0.5),),
            bond_overrides=(AtomBondOverride(1, "O", "H", "O", OverrideVisibility.HIDE),),
            hydrogen_bond_overrides=(AtomHydrogenBondOverride(0, "O", OverrideVisibility.HIDE),),
        ),
    )
    baseline = replace(
        _default_state().style,
        size_profiles=SizeProfileSettings(
            covalent=CovalentSizeProfile(global_scale=0.8),
        ),
        export=ExportSettings("png", 300, False),
    )

    reset = reset_visual_modules_from_style(current, baseline, atoms)

    assert reset.style.view is current.style.view
    assert reset.style.export is current.style.export
    assert reset.style.atom_cell == baseline.atom_cell
    assert reset.style.size_profiles == baseline.size_profiles
    assert reset.style.bonds == merge_pair_rules_for_structure(atoms, baseline.bonds)
    assert reset.style.cell_periodic == normalize_periodic_settings(
        atoms, baseline.cell_periodic
    )
    assert reset.atom_selection == AtomSelectionSettings()


def test_reset_visual_modules_is_atomic_for_invalid_periodic_baseline():
    atoms = Atoms("H" * 1001, positions=[[0, 0, 0]] * 1001, cell=[5, 5, 5], pbc=True)
    current = _default_state()
    baseline = replace(
        current.style,
        cell_periodic=CellPeriodicSettings(
            a=PeriodicRange(0, 5), b=PeriodicRange(0, 5), c=PeriodicRange(0, 2)
        ),
    )

    started = perf_counter()
    with pytest.raises(ValueError, match="50,000"):
        reset_visual_modules_from_style(current, baseline, atoms)
    assert perf_counter() - started < 1.0

    assert current.style is current.style
    assert current.atom_selection is current.atom_selection


def test_periodic_ranges_normalize_non_pbc_axes_and_enforce_instance_limit():
    atoms = Atoms(
        "H2",
        positions=[[0, 0, 0], [0.7, 0, 0]],
        cell=[5, 6, 7],
        pbc=[True, False, True],
    )
    settings = CellPeriodicSettings(
        a=PeriodicRange(-1, 2),
        b=PeriodicRange(-4, 5),
        c=PeriodicRange(0, 2),
    )
    normalized = normalize_periodic_settings(atoms, settings)
    assert normalized.b == PeriodicRange(0, 1)
    assert estimate_periodic_atom_instances(atoms, normalized) == 12

    too_large = Atoms(
        "H" * 1001,
        positions=np.zeros((1001, 3)),
        cell=[5, 5, 5],
        pbc=True,
    )
    with pytest.raises(ValueError, match="50,000"):
        normalize_periodic_settings(
            too_large,
            CellPeriodicSettings(
                a=PeriodicRange(0, 5),
                b=PeriodicRange(0, 5),
                c=PeriodicRange(0, 2),
            ),
        )


def test_apply_camera_only_does_not_commit_other_modules():
    default_state = _default_state()

    changed = apply_camera_only(
        default_state,
        CameraState(eye=(0.0, 2.0, 0.0)),
    )

    assert changed.style.view.camera.eye == (0.0, 2.0, 0.0)
    assert changed.style.atom_cell is default_state.style.atom_cell
    assert changed.style.bonds is default_state.style.bonds
    assert changed.atom_selection is default_state.atom_selection
    assert changed.style.export is default_state.style.export


def test_replacing_atom_cell_does_not_change_the_other_four_modules():
    state = _default_state()
    replacement = replace(
        state.style.atom_cell,
        outline_width=0.75,
    )

    changed = replace_atom_cell(state, replacement)

    assert changed.style.atom_cell == replacement
    assert changed.style.view is state.style.view
    assert changed.style.bonds is state.style.bonds
    assert changed.atom_selection is state.atom_selection
    assert changed.style.export is state.style.export


def test_portable_style_replacement_preserves_specific_atom_state():
    atoms = Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]])
    current = replace(
        _default_state(),
        atom_selection=AtomSelectionSettings(
            selected_atom_indices=(1,),
            color_overrides=(AtomColorOverride(1, "O", "#336699"),),
        ),
    )
    imported_style = replace(
        current.style,
        size_profiles=SizeProfileSettings(
            covalent=CovalentSizeProfile(global_scale=0.8),
        ),
        bonds=BondModuleSettings(),
    )

    updated = apply_portable_style(current, imported_style, atoms)

    assert updated.atom_selection == current.atom_selection
    assert updated.style == merge_portable_style_for_structure(imported_style, atoms)


def test_explicit_pair_wins_missing_pairs_generate_and_zero_match_survives():
    atoms = Atoms(
        "NaClCaO",
        positions=[[0, 0, 0], [2.3, 0, 0], [10, 0, 0], [12.3, 0, 0]],
    )
    explicit_ca_o = BondPairRule("Ca", "O", 2.1, 2.8, enabled=False)
    explicit_si_o = BondPairRule("Si", "O", 1.4, 1.9, enabled=True)
    bonds = BondModuleSettings(pair_rules=(explicit_ca_o, explicit_si_o))

    merged = merge_pair_rules_for_structure(atoms, bonds)
    by_pair = {rule.pair: rule for rule in merged.pair_rules}

    assert by_pair[("Ca", "O")] is explicit_ca_o
    assert by_pair[("Ca", "O")].enabled is False
    assert ("Cl", "Na") in by_pair
    assert by_pair[("Cl", "Na")].min_distance == 0.0
    assert ("O", "Si") in by_pair


def test_generated_pair_defaults_use_actual_minimum_and_strict_two_angstrom():
    atoms = Atoms(
        "H2CaO",
        positions=[[0, 0, 0], [2.0, 0, 0], [10, 0, 0], [12.1, 0, 0]],
    )
    defaults = PairRuleDefaults(
        long_distance_threshold_angstrom=2.0,
        pair_distance_multipliers=(("H", "H", 4.0), ("Ca", "O", 1.2)),
    )
    merged = merge_pair_rules_for_structure(
        atoms,
        BondModuleSettings(defaults=defaults),
    )
    by_pair = {rule.pair: rule for rule in merged.pair_rules}
    assert by_pair[("H", "H")].enabled is True
    assert by_pair[("H", "H")].participates_in_periodic_unwrap is True
    assert by_pair[("Ca", "O")].enabled is False
    assert by_pair[("Ca", "O")].participates_in_periodic_unwrap is False


def test_explicit_pair_flags_are_never_reclassified_from_structure_distance():
    explicit = BondPairRule(
        "Ca", "O", 0.0, 2.8,
        enabled=True,
        participates_in_periodic_unwrap=False,
    )
    atoms = Atoms("CaO", positions=[[0, 0, 0], [2.3, 0, 0]])
    merged = merge_pair_rules_for_structure(
        atoms,
        BondModuleSettings(pair_rules=(explicit,)),
    )
    assert merged.pair_rules == (explicit,)


def test_render_context_runs_one_distance_match_for_generated_pairs(monkeypatch):
    atoms = Atoms("CaO", positions=[[0, 0, 0], [2.3, 0, 0]])
    calls = 0
    original = visual_state_module.resolve_bonds

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(visual_state_module, "resolve_bonds", counted)
    context = resolve_render_context(atoms, VisualizationState())
    assert calls == 1
    assert context.bond_resolution.matched
    assert context.bond_resolution.visible == ()


def test_render_context_resolves_specific_atom_style_and_bond_override_once():
    atoms = Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]])
    state = replace(
        _default_state(),
        atom_selection=AtomSelectionSettings(
            color_overrides=(AtomColorOverride(1, "O", "#336699"),),
            color_strengths=(AtomColorStrength(1, "O", 0.5),),
            bond_overrides=(
                AtomBondOverride(
                    1,
                    "O",
                    "H",
                    "O",
                    OverrideVisibility.HIDE,
                ),
            ),
        ),
    )

    context = resolve_render_context(atoms, state)

    assert context.config.atom_color_overrides == {1: "#336699"}
    assert context.config.atom_color_strengths == {1: 0.5}
    assert context.config.get_atom_colors(["H", "O"]) == [
        "#E6E6E5",
        "#99B2CC",
    ]
    assert context.bond_settings.atom_overrides == state.atom_selection.bond_overrides


def test_render_context_carries_one_bond_resolution_and_periodic_display():
    """缺少最终 matched 拓扑或周期实例时，统一上下文契约必须失败。"""
    atoms = Atoms(
        "HHO",
        scaled_positions=[[0.01, 0.5, 0.5], [0.92, 0.5, 0.5], [0.95, 0.5, 0.5]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    state = replace(
        _default_state(),
        style=replace(
            _default_state().style,
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=0,
                a=PeriodicRange(-1, 2),
            ),
        ),
    )

    context = resolve_render_context(atoms, state)

    assert context.bond_resolution.matched
    assert len(context.periodic_display.replica_translations) == 3
    assert context.hidden_atom_indices == frozenset()


def test_render_context_uses_topology_flag_not_display_visibility_for_unwrap():
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.6, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )

    def context_for(rule):
        return resolve_render_context(
            atoms,
            VisualizationState(
                style=PortableStyle(
                    bonds=BondModuleSettings(pair_rules=(rule,)),
                    cell_periodic=CellPeriodicSettings(a=PeriodicRange(-1, 2)),
                )
            ),
        )

    base_rule = BondPairRule("H", "H", 0.0, 1.0)
    visible = context_for(base_rule)
    display_hidden = context_for(replace(base_rule, enabled=False))
    topology_excluded = context_for(
        replace(base_rule, enabled=False, participates_in_periodic_unwrap=False)
    )

    assert display_hidden.periodic_display.base_image_shifts == (
        visible.periodic_display.base_image_shifts
    )
    assert {
        bond.bond_id for bond in display_hidden.periodic_topology_bonds
    } == {bond.bond_id for bond in visible.periodic_topology_bonds}
    assert {
        item.source_bond.bond_id
        for item in display_hidden.periodic_display.bond_instances
    } == {item.source_bond.bond_id for item in visible.periodic_display.bond_instances}
    assert topology_excluded.periodic_topology_bonds == ()
    assert topology_excluded.periodic_display.base_image_shifts == ((0, 0, 0),) * 2
    assert topology_excluded.periodic_display.base_image_shifts != (
        visible.periodic_display.base_image_shifts
    )


def test_context_is_not_resolved_again_by_either_2d_entry_point(monkeypatch):
    """已有上下文再次解析键或周期图会破坏单一实例身份来源。"""
    atoms = Atoms(
        "HO",
        scaled_positions=[[0.02, 0.5, 0.5], [0.98, 0.5, 0.5]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    state = replace(
        _default_state(),
        style=replace(
            _default_state().style,
            cell_periodic=CellPeriodicSettings(show_unit_cell=0),
        ),
    )
    calls = {"bonds": 0, "display": 0}
    real_resolve_bonds = visual_state_module.resolve_bonds
    real_build_display = periodic_display_module.build_periodic_display

    def counted_resolve_bonds(*args, **kwargs):
        calls["bonds"] += 1
        return real_resolve_bonds(*args, **kwargs)

    def counted_build_display(*args, **kwargs):
        calls["display"] += 1
        return real_build_display(*args, **kwargs)

    for module in (visual_state_module, view_module, pipeline_module):
        monkeypatch.setattr(
            module,
            "resolve_bonds",
            counted_resolve_bonds,
            raising=False,
        )
        monkeypatch.setattr(
            module,
            "build_periodic_display",
            counted_build_display,
            raising=False,
        )

    context = resolve_render_context(atoms, state)
    assert calls == {"bonds": 1, "display": 1}

    figure_from_view = render_2d(atoms, context.config, render_context=context)
    figure_from_pipeline = render_atoms(atoms, render_context=context)

    assert calls == {"bonds": 1, "display": 1}
    plt.close(figure_from_view)
    plt.close(figure_from_pipeline)


def test_render_context_can_drive_both_legacy_2d_and_3d_entry_points():
    atoms = Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]])
    state = replace(
        _default_state(),
        style=replace(
            _default_state().style,
            size_profiles=SizeProfileSettings(
                covalent=CovalentSizeProfile(
                    global_scale=0.75,
                    bond_width_ratio=0.30,
                ),
            ),
            bonds=replace(
                _default_state().style.bonds,
                style=BondStrokeStyle(
                    stroke_width=0.10,
                    stroke_color="#231815",
                ),
            ),
        ),
    )
    context = resolve_render_context(atoms, state)

    figure_2d = render_2d(
        atoms,
        context.config,
        render_context=context,
    )
    figure_3d = create_3d_figure(
        atoms,
        context.config,
        render_context=context,
    )

    circles = [
        patch for patch in figure_2d.axes[0].patches if type(patch) is Circle
    ]
    atom_trace = next(trace for trace in figure_3d.data if trace.name == "原子")
    assert circles[0].get_radius() == 0.75 * 0.31
    assert np.isclose(atom_trace.marker.size[0], 0.75 * 0.31 * 15)
    plt.close(figure_2d)
