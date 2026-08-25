"""Structure-topology cache keys and composition boundaries."""

from dataclasses import replace

import pytest
from ase import Atoms

from meia.atom_styles import (
    AtomColorStrength,
    AtomHydrogenBondOverride,
    AtomSelectionSettings,
    HiddenAtom,
)
from meia.bond_rules import AtomBondOverride, BondPairRule, OverrideVisibility
from meia.hydrogen_bonds import HydrogenBondSettings
from meia.periodic_display import CellPeriodicSettings, PeriodicRange
from meia.render_topology import (
    build_render_topology,
    compose_render_context,
    topology_key,
)
from meia.view_state import CameraState
from meia.visual_state import (
    AtomCellSettings,
    BondModuleSettings,
    ExportSettings,
    PortableStyle,
    ViewSettings,
    VisualizationState,
)


def _fixture() -> tuple[Atoms, VisualizationState]:
    atoms = Atoms(
        "OHO",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [2.8, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    state = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.7, 1.2),),
                hydrogen_bonds=HydrogenBondSettings(
                    draw=True,
                    max_hydrogen_oxygen_distance=2.5,
                    min_angle_degrees=120.0,
                ),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=2,
                a=PeriodicRange(0, 1),
                b=PeriodicRange(0, 1),
                c=PeriodicRange(0, 1),
            ),
        )
    )
    return atoms, state


def _replace_style(state: VisualizationState, **changes) -> VisualizationState:
    return replace(state, style=replace(state.style, **changes))


def test_topology_key_ignores_style_only_changes():
    atoms, state = _fixture()
    baseline = topology_key(atoms, state, structure_id="fixture")

    colors = _replace_style(
        state,
        atom_cell=AtomCellSettings(element_colors={"O": "#00FF00"}),
    )
    strengths = replace(
        state,
        atom_selection=AtomSelectionSettings(
            default_color_strength=0.3,
            color_strengths=(AtomColorStrength(0, "O", 1.0),),
        ),
    )
    current_profile = state.style.size_profiles.covalent
    sizes = _replace_style(
        state,
        size_profiles=replace(
            state.style.size_profiles,
            covalent=replace(
                current_profile,
                global_scale=0.8,
                bond_width_ratio=0.7,
            ),
        ),
    )
    exports = _replace_style(
        state,
        export=ExportSettings(format="png", dpi=300, transparent=False),
    )
    camera = _replace_style(
        state,
        view=ViewSettings(camera=CameraState(eye=(2.0, 1.0, 1.0))),
    )
    cell_layer = _replace_style(
        state,
        cell_periodic=replace(state.style.cell_periodic, show_unit_cell=0),
    )

    for changed in (colors, strengths, sizes, exports, camera, cell_layer):
        assert topology_key(atoms, changed, structure_id="fixture") == baseline


def test_topology_key_changes_for_structure_and_connectivity_inputs():
    atoms, state = _fixture()
    baseline = topology_key(atoms, state, structure_id="fixture")
    rule = state.style.bonds.pair_rules[0]

    moved = atoms.copy()
    moved.positions[1, 0] += 0.05
    pair_distance = _replace_style(
        state,
        bonds=replace(
            state.style.bonds,
            pair_rules=(replace(rule, max_distance=1.3),),
        ),
    )
    periodic_range = _replace_style(
        state,
        cell_periodic=replace(
            state.style.cell_periodic,
            a=PeriodicRange(-1, 2),
        ),
    )
    unwrap = _replace_style(
        state,
        bonds=replace(
            state.style.bonds,
            pair_rules=(replace(rule, participates_in_periodic_unwrap=False),),
        ),
    )
    hidden = replace(
        state,
        atom_selection=AtomSelectionSettings(hidden_atoms=(HiddenAtom(2, "O"),)),
    )
    bond_override = replace(
        state,
        atom_selection=AtomSelectionSettings(
            bond_overrides=(
                AtomBondOverride(
                    0,
                    "O",
                    "H",
                    "O",
                    OverrideVisibility.HIDE,
                ),
            ),
        ),
    )
    hydrogen_override = replace(
        state,
        atom_selection=AtomSelectionSettings(
            hydrogen_bond_overrides=(
                AtomHydrogenBondOverride(2, "O", OverrideVisibility.HIDE),
            ),
        ),
    )
    hydrogen_distance = _replace_style(
        state,
        bonds=replace(
            state.style.bonds,
            hydrogen_bonds=replace(
                state.style.bonds.hydrogen_bonds,
                max_hydrogen_oxygen_distance=2.2,
            ),
        ),
    )
    hydrogen_angle = _replace_style(
        state,
        bonds=replace(
            state.style.bonds,
            hydrogen_bonds=replace(
                state.style.bonds.hydrogen_bonds,
                min_angle_degrees=150.0,
            ),
        ),
    )

    changed_inputs = (
        (moved, state),
        (atoms, pair_distance),
        (atoms, periodic_range),
        (atoms, unwrap),
        (atoms, hidden),
        (atoms, bond_override),
        (atoms, hydrogen_override),
        (atoms, hydrogen_distance),
        (atoms, hydrogen_angle),
    )
    for changed_atoms, changed_state in changed_inputs:
        assert (
            topology_key(changed_atoms, changed_state, structure_id="fixture")
            != baseline
        )


def test_composition_reuses_topology_but_rejects_stale_connectivity():
    atoms, state = _fixture()
    topology = build_render_topology(atoms, state, structure_id="fixture")
    recolored = _replace_style(
        state,
        atom_cell=AtomCellSettings(element_colors={"O": "#00FF00"}),
    )

    context = compose_render_context(
        atoms,
        recolored,
        topology,
        structure_id="fixture",
    )
    assert context.periodic_display is topology.periodic_display
    assert context.config.get_atom_colors(["O"])[0] == "#00FF00"

    changed = _replace_style(
        state,
        cell_periodic=replace(
            state.style.cell_periodic,
            a=PeriodicRange(0, 2),
        ),
    )
    with pytest.raises(ValueError, match="topology"):
        compose_render_context(
            atoms,
            changed,
            topology,
            structure_id="fixture",
        )
