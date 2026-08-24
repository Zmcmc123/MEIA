"""元素无关的周期成键图引擎回归测试。"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms

from meia.bond_rules import ResolvedBond
from meia.periodic_display import (
    CellPeriodicSettings,
    PeriodicRange,
    build_periodic_display,
    normalize_periodic_settings,
)
from meia.i18n import I18n, Locale


def test_periodic_range_and_instance_limit_have_exact_english_diagnostics():
    with pytest.raises(ValueError) as range_error:
        PeriodicRange(1, 1)
    assert I18n(Locale.EN).error_text(
        range_error.value, "periodic.apply_failed"
    ) == "The periodic range end (1) must be greater than its start (1)."

    atoms = Atoms("HH", positions=np.zeros((2, 3)), cell=[10, 10, 10], pbc=True)
    with pytest.raises(ValueError) as limit_error:
        normalize_periodic_settings(
            atoms,
            CellPeriodicSettings(a=PeriodicRange(0, 25_001)),
        )
    assert I18n(Locale.EN).error_text(
        limit_error.value, "periodic.apply_failed"
    ) == (
        "The periodic display would create 50,002 atom instances, "
        "exceeding the limit of 50,000."
    )


def resolved(i, j, offset, symbol_i, symbol_j):
    offset_text = "_".join(str(value) for value in offset)
    return ResolvedBond(
        i=i,
        j=j,
        offset=offset,
        distance=1.0,
        pair=tuple(sorted((symbol_i, symbol_j))),
        bond_id=f"bond_{i}_{j}_{offset_text}",
        visible=True,
        visibility_source="pair_enabled",
    )


def test_finite_component_uses_graph_center_and_keeps_split_molecule_whole():
    atoms = Atoms(
        "HHO",
        scaled_positions=[[0.01, 0.5, 0.5], [0.92, 0.5, 0.5], [0.95, 0.5, 0.5]],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = (
        resolved(0, 2, (-1, 0, 0), "H", "O"),
        resolved(1, 2, (0, 0, 0), "H", "O"),
    )

    display = build_periodic_display(atoms, bonds, CellPeriodicSettings())

    assert display.base_image_shifts == ((1, 0, 0), (0, 0, 0), (0, 0, 0))
    assert (
        np.linalg.norm(
            display.atom_by_key[(0, (0, 0, 0))].position
            - display.atom_by_key[(2, (0, 0, 0))].position
        )
        < 1.2
    )


def test_instance_positions_use_full_triclinic_cell_matrix():
    cell = np.array([[4.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.3, 6.0]])
    atoms = Atoms("He", positions=[[0.2, 0.3, 0.4]], cell=cell, pbc=True)
    settings = CellPeriodicSettings(
        a=PeriodicRange(-1, 1),
        b=PeriodicRange(0, 2),
    )

    display = build_periodic_display(atoms, (), settings)

    instance = display.atom_by_key[(0, (-1, 1, 0))]
    assert np.allclose(
        instance.position,
        atoms.positions[0] + np.dot((-1, 1, 0), cell),
    )


def test_periodic_core_stays_primary_while_bridge_branch_unwraps():
    atoms = Atoms(
        "C4",
        scaled_positions=[
            [0.20, 0.20, 0.20],
            [0.40, 0.20, 0.20],
            [0.30, 0.40, 0.20],
            [0.20, 0.20, 0.95],
        ],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = (
        resolved(0, 1, (0, 0, 0), "C", "C"),
        resolved(1, 2, (0, 0, 0), "C", "C"),
        resolved(0, 2, (0, 0, 1), "C", "C"),
        resolved(0, 3, (0, 0, -1), "C", "C"),
    )

    display = build_periodic_display(atoms, bonds, CellPeriodicSettings())

    assert display.base_image_shifts[:3] == ((0, 0, 0),) * 3
    assert display.base_image_shifts[3] == (0, 0, -1)
    assert not display.diagnostics


def test_two_conflicting_nonbridge_attachments_stay_primary_with_diagnostic():
    atoms = Atoms(
        "C4",
        scaled_positions=[
            [0.20, 0.20, 0.20],
            [0.40, 0.20, 0.20],
            [0.30, 0.40, 0.20],
            [0.30, 0.20, 0.95],
        ],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = (
        resolved(0, 1, (0, 0, 0), "C", "C"),
        resolved(1, 2, (0, 0, 0), "C", "C"),
        resolved(0, 2, (0, 0, 1), "C", "C"),
        resolved(0, 3, (0, 0, -1), "C", "C"),
        resolved(1, 3, (0, 0, 0), "C", "C"),
    )

    display = build_periodic_display(atoms, bonds, CellPeriodicSettings())

    assert display.base_image_shifts[3] == (0, 0, 0)
    diagnostic = next(
        item
        for item in display.diagnostics
        if item.code == "ambiguous_periodic_attachment"
    )
    assert 3 in diagnostic.atom_indices


def test_parallel_edges_form_periodic_block_instead_of_false_bridges():
    atoms = Atoms(
        "C2",
        scaled_positions=[[0.2, 0.2, 0.2], [0.4, 0.2, 0.2]],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = (
        resolved(0, 1, (0, 0, 0), "C", "C"),
        resolved(0, 1, (0, 0, 1), "C", "C"),
    )

    display = build_periodic_display(atoms, bonds, CellPeriodicSettings())

    assert display.base_image_shifts == ((0, 0, 0), (0, 0, 0))
    assert not display.diagnostics


def test_periodic_self_edge_connects_two_requested_images():
    atoms = Atoms(
        "C",
        scaled_positions=[[0.2, 0.2, 0.2]],
        cell=[10, 10, 10],
        pbc=True,
    )
    bond = resolved(0, 0, (1, 0, 0), "C", "C")
    settings = CellPeriodicSettings(a=PeriodicRange(0, 2))

    display = build_periodic_display(atoms, (bond,), settings)

    assert display.base_image_shifts == ((0, 0, 0),)
    assert len(display.bond_instances) == 1
    instance = display.bond_instances[0]
    assert instance.atom_i_key == (0, (0, 0, 0))
    assert instance.atom_j_key == (0, (1, 0, 0))


def test_disabled_unwrapping_keeps_zero_shifts_and_uses_matched_hidden_bond():
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.6, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    hidden = resolved(0, 1, (-1, 0, 0), "H", "H")
    hidden = ResolvedBond(
        **{
            **hidden.__dict__,
            "visible": False,
            "visibility_source": "pair_disabled",
        }
    )
    positions = atoms.positions.copy()
    cell = atoms.cell.array.copy()
    pbc = atoms.pbc.copy()
    symbols = atoms.get_chemical_symbols()
    settings = CellPeriodicSettings(
        unwrap_bonded_groups=False,
        a=PeriodicRange(0, 2),
    )

    display = build_periodic_display(atoms, (hidden,), settings)

    assert display.base_image_shifts == ((0, 0, 0), (0, 0, 0))
    assert not display.diagnostics
    assert len(display.bond_instances) == 1
    instance = display.bond_instances[0]
    assert instance.atom_i_key == (0, (1, 0, 0))
    assert instance.atom_j_key == (1, (0, 0, 0))
    assert np.array_equal(atoms.positions, positions)
    assert np.array_equal(atoms.cell.array, cell)
    assert np.array_equal(atoms.pbc, pbc)
    assert atoms.get_chemical_symbols() == symbols


def test_replica_and_instance_order_is_deterministic_for_reordered_bonds():
    atoms = Atoms(
        "C3",
        scaled_positions=[[0.1, 0.1, 0.1], [0.2, 0.1, 0.1], [0.3, 0.1, 0.1]],
        cell=[10, 10, 10],
        pbc=True,
    )
    bonds = (
        resolved(1, 2, (0, 0, 0), "C", "C"),
        resolved(0, 1, (0, 0, 0), "C", "C"),
    )
    settings = CellPeriodicSettings(
        a=PeriodicRange(-1, 1),
        b=PeriodicRange(0, 2),
    )

    first = build_periodic_display(atoms, bonds, settings)
    second = build_periodic_display(atoms, tuple(reversed(bonds)), settings)

    assert first.replica_translations == (
        (-1, 0, 0),
        (-1, 1, 0),
        (0, 0, 0),
        (0, 1, 0),
    )
    assert [item.instance_id for item in first.atom_instances] == [
        item.instance_id for item in second.atom_instances
    ]
    assert [item.bond_instance_id for item in first.bond_instances] == [
        item.bond_instance_id for item in second.bond_instances
    ]


@pytest.mark.parametrize(
    "bond",
    [
        resolved(0, 2, (0, 0, 0), "H", "H"),
        resolved(0, 1, (0, 0), "H", "H"),
        resolved(0, 1, (0, True, 0), "H", "H"),
    ],
)
def test_invalid_bond_indices_and_offsets_are_rejected(bond):
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])

    with pytest.raises((TypeError, ValueError)):
        build_periodic_display(atoms, (bond,), CellPeriodicSettings())


def test_bridge_detection_handles_chain_beyond_python_recursion_limit():
    atom_count = 1_200
    atoms = Atoms("C" * atom_count, positions=np.zeros((atom_count, 3)))
    bonds = tuple(
        resolved(index, index + 1, (0, 0, 0), "C", "C")
        for index in range(atom_count - 1)
    )

    display = build_periodic_display(atoms, bonds, CellPeriodicSettings())

    assert display.base_image_shifts == ((0, 0, 0),) * atom_count


def test_duplicate_source_bond_ids_still_produce_unique_display_ids():
    atoms = Atoms(
        "C4",
        positions=[[0, 0, 0], [1, 0, 0], [3, 0, 0], [4, 0, 0]],
    )
    first = replace(
        resolved(0, 1, (0, 0, 0), "C", "C"),
        bond_id="duplicate",
    )
    second = replace(
        resolved(2, 3, (0, 0, 0), "C", "C"),
        bond_id="duplicate",
    )

    display = build_periodic_display(
        atoms,
        (first, first, second),
        CellPeriodicSettings(),
    )

    assert len(display.bond_instances) == 2
    instance_ids = [item.bond_instance_id for item in display.bond_instances]
    assert len(set(instance_ids)) == 2
    assert any("__source_0_1_offset_0_0_0__" in value for value in instance_ids)
    assert any("__source_2_3_offset_0_0_0__" in value for value in instance_ids)


def test_bond_instance_dedupe_keeps_different_source_ids_for_same_endpoints():
    atoms = Atoms("C2", positions=[[0, 0, 0], [1, 0, 0]])
    first = replace(
        resolved(0, 1, (0, 0, 0), "C", "C"),
        bond_id="first",
    )
    duplicate_first = replace(first)
    second = replace(first, bond_id="second")

    display = build_periodic_display(
        atoms,
        (first, duplicate_first, second),
        CellPeriodicSettings(),
    )

    assert len(display.bond_instances) == 2
    assert {item.source_bond.bond_id for item in display.bond_instances} == {
        "first",
        "second",
    }


def _theta_bonds(path_offsets):
    bonds = []
    for middle_index, offset in enumerate(path_offsets, start=1):
        bonds.extend(
            (
                resolved(0, middle_index, (0, 0, 0), "C", "C"),
                resolved(middle_index, 4, offset, "C", "C"),
            )
        )
    return tuple(bonds)


def mixed_theta_bonds(path_offsets):
    bonds = []
    for oxygen_index, offset in enumerate(path_offsets, start=1):
        bonds.extend(
            (
                resolved(0, oxygen_index, offset, "Si", "O"),
                resolved(oxygen_index, 4, (0, 0, 0), "O", "Ca"),
            )
        )
    return tuple(bonds)


def test_display_edges_do_not_change_base_shifts_when_excluded_from_topology():
    atoms = Atoms("SiO3Ca", positions=np.zeros((5, 3)), cell=[10, 10, 10], pbc=True)
    all_bonds = mixed_theta_bonds(((0, 0, 0), (0, 0, 0), (0, 0, 1)))
    si_o = tuple(bond for bond in all_bonds if bond.pair == ("O", "Si"))
    display = build_periodic_display(
        atoms,
        all_bonds,
        CellPeriodicSettings(c=PeriodicRange(0, 2)),
        topology_bonds=si_o,
    )
    assert not display.diagnostics
    assert display.base_image_shifts[3] == (0, 0, 1)
    assert {item.source_bond.pair for item in display.bond_instances} >= {
        ("Ca", "O"),
        ("O", "Si"),
    }


def test_mixed_pair_ambiguity_reports_actionable_element_pairs():
    atoms = Atoms("SiO3Ca", positions=np.zeros((5, 3)), cell=[10, 10, 10], pbc=True)
    bonds = mixed_theta_bonds(((0, 0, 0), (0, 0, 0), (0, 0, 1)))
    display = build_periodic_display(atoms, bonds, CellPeriodicSettings())
    assert display.diagnostics[0].conflicting_element_pairs == (
        ("Ca", "O"),
        ("O", "Si"),
    )


def test_theta_ambiguity_is_invariant_when_periodic_path_is_renumbered():
    atoms = Atoms("C5", positions=np.zeros((5, 3)))
    periodic_last = build_periodic_display(
        atoms,
        _theta_bonds(((0, 0, 0), (0, 0, 0), (0, 0, 1))),
        CellPeriodicSettings(),
    )
    periodic_first = build_periodic_display(
        atoms,
        _theta_bonds(((0, 0, 1), (0, 0, 0), (0, 0, 0))),
        CellPeriodicSettings(),
    )

    diagnostics = tuple(
        tuple(item.code for item in display.diagnostics)
        for display in (periodic_last, periodic_first)
    )
    assert diagnostics == (
        ("ambiguous_periodic_attachment",),
        ("ambiguous_periodic_attachment",),
    )
    assert periodic_last.base_image_shifts == ((0, 0, 0),) * 5
    assert periodic_first.base_image_shifts == ((0, 0, 0),) * 5


def test_full_rank_theta_periodic_cycles_are_not_ambiguous():
    atoms = Atoms("C5", positions=np.zeros((5, 3)))
    display = build_periodic_display(
        atoms,
        _theta_bonds(((0, 0, 0), (1, 0, 0), (0, 1, 0))),
        CellPeriodicSettings(),
    )

    assert display.base_image_shifts == ((0, 0, 0),) * 5
    assert not display.diagnostics
