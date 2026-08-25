"""Public CONTCAR-derived large-system example and workspace snapshot."""

from collections import Counter
import hashlib
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from ase.data import covalent_radii
from ase.neighborlist import neighbor_list

from meia.display_complexity import measure_display_complexity
from meia.io import read_structure
from meia.presets import WorkspaceSnapshot, parse_preset
from meia.render_topology import build_render_topology, compose_render_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "scripts" / "generate_large_system_example.py"
BASE_STRUCTURE = PROJECT_ROOT / "examples" / "CONTCAR"
STRUCTURE = PROJECT_ROOT / "examples" / "MEIA_large_system_5x5_200H2O.xyz"
SNAPSHOT = (
    PROJECT_ROOT
    / "examples"
    / "MEIA_large_system_5x5_200H2O.workspace.meia.json"
)
OLD_STRUCTURE = PROJECT_ROOT / "examples" / "MEIA_large_slab_water_6246.xyz"
OLD_SNAPSHOT = (
    PROJECT_ROOT
    / "examples"
    / "MEIA_large_slab_water_6246.workspace.meia.json"
)
BASE_SHA256 = "187ee6a6d1c5bffc2b55a8ea254f0dc86c82a1f56743fcdad55504488b399d5f"
REPEATED_ATOM_COUNT = 5 * 5 * 225
ADDED_WATER_COUNT = 200
TOTAL_ATOM_COUNT = REPEATED_ATOM_COUNT + 3 * ADDED_WATER_COUNT


def test_large_system_example_is_reproducible_and_derived_from_contcar():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256(BASE_STRUCTURE.read_bytes()).hexdigest() == BASE_SHA256
    assert not OLD_STRUCTURE.exists()
    assert not OLD_SNAPSHOT.exists()

    atoms = read_structure(str(STRUCTURE))
    snapshot = parse_preset(SNAPSHOT.read_bytes())
    assert isinstance(snapshot, WorkspaceSnapshot)
    assert len(atoms) == TOTAL_ATOM_COUNT == 6225
    assert Counter(atoms.get_chemical_symbols()) == {
        "H": 1300,
        "C": 25,
        "O": 3100,
        "Si": 600,
        "Ca": 1200,
    }
    np.testing.assert_allclose(
        atoms.cell.lengths(),
        [67.825385, 51.10115, 32.73259],
        atol=1e-8,
    )
    assert len(snapshot.structure.symbols) == len(atoms)
    assert snapshot.structure.source_name == STRUCTURE.name
    assert snapshot.state.atom_selection.selected_atom_indices == (
        REPEATED_ATOM_COUNT,
        REPEATED_ATOM_COUNT + 1,
        REPEATED_ATOM_COUNT + 2,
    )
    periodic = snapshot.state.style.cell_periodic
    assert (periodic.a.start, periodic.a.end) == (0, 1)
    assert (periodic.b.start, periodic.b.end) == (0, 1)
    assert (periodic.c.start, periodic.c.end) == (0, 1)
    topology = build_render_topology(
        atoms,
        snapshot.state,
        structure_id="large-example",
    )
    context = compose_render_context(
        atoms,
        snapshot.state,
        topology,
        structure_id="large-example",
    )
    complexity = measure_display_complexity(len(atoms), context)
    assert complexity.atom_instance_count == TOTAL_ATOM_COUNT
    assert complexity.manual_2d_recommended is True
    assert complexity.large_3d_interaction is True
    assert complexity.extreme_3d_interaction is False


def test_added_water_geometry_has_no_unintended_compressed_contacts():
    atoms = read_structure(str(STRUCTURE))
    tail = atoms[REPEATED_ATOM_COUNT:]
    symbols = tail.get_chemical_symbols()
    assert len(tail) == 3 * ADDED_WATER_COUNT
    assert all(
        symbols[offset : offset + 3] == ["O", "H", "H"]
        for offset in range(0, len(symbols), 3)
    )

    for offset in range(0, len(tail), 3):
        water = tail[offset : offset + 3]
        distances = water.get_all_distances(mic=True)
        assert distances[0, 1] == pytest.approx(0.9572, abs=1e-6)
        assert distances[0, 2] == pytest.approx(0.9572, abs=1e-6)
        assert distances[1, 2] == pytest.approx(1.5139, abs=1e-4)

    indices_i, indices_j, distances = neighbor_list("ijd", atoms, 2.2)
    failures = []
    for atom_i, atom_j, distance in zip(indices_i, indices_j, distances):
        atom_i = int(atom_i)
        atom_j = int(atom_j)
        if atom_i >= atom_j or max(atom_i, atom_j) < REPEATED_ATOM_COUNT:
            continue
        same_added_water = (
            atom_i >= REPEATED_ATOM_COUNT
            and atom_j >= REPEATED_ATOM_COUNT
            and (atom_i - REPEATED_ATOM_COUNT) // 3
            == (atom_j - REPEATED_ATOM_COUNT) // 3
        )
        if same_added_water:
            continue
        lower_bound = 0.95 * (
            covalent_radii[atoms.numbers[atom_i]]
            + covalent_radii[atoms.numbers[atom_j]]
        )
        if float(distance) < float(lower_bound):
            failures.append((atom_i, atom_j, float(distance), float(lower_bound)))
    assert failures == []
