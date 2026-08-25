#!/usr/bin/env python3
"""Generate the public CONTCAR-derived large-system test workspace."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import io
from pathlib import Path
import os
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
BASE_STRUCTURE = EXAMPLES_DIR / "CONTCAR"
BASE_STRUCTURE_SHA256 = (
    "187ee6a6d1c5bffc2b55a8ea254f0dc86c82a1f56743fcdad55504488b399d5f"
)
STRUCTURE_NAME = "MEIA_large_system_5x5_200H2O.xyz"
SNAPSHOT_NAME = "MEIA_large_system_5x5_200H2O.workspace.meia.json"
CREATED_AT = "2026-08-25T00:00:00+08:00"
SUPERCELL = (5, 5, 1)
WATER_GRID = (10, 10)
WATER_LAYER_Z = (25.2, 28.6)
WATER_OH_DISTANCE = 0.9572
WATER_HOH_ANGLE_DEGREES = 104.52


def _read_verified_base_structure():
    from ase.io import read

    payload = BASE_STRUCTURE.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != BASE_STRUCTURE_SHA256:
        raise RuntimeError(
            "examples/CONTCAR changed; review the large-example construction "
            f"before updating its expected SHA-256 (found {digest})"
        )
    atoms = read(BASE_STRUCTURE)
    if len(atoms) != 225:
        raise RuntimeError(f"expected 225 atoms in examples/CONTCAR, found {len(atoms)}")
    return atoms


def _water_positions(cell_lengths: np.ndarray) -> list[tuple[float, float, float]]:
    length_x, length_y, _ = (float(value) for value in cell_lengths)
    grid_x, grid_y = WATER_GRID
    step_x = length_x / grid_x
    step_y = length_y / grid_y
    theta = np.deg2rad(WATER_HOH_ANGLE_DEGREES)
    reference = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [WATER_OH_DISTANCE, 0.0, 0.0],
            [
                WATER_OH_DISTANCE * np.cos(theta),
                WATER_OH_DISTANCE * np.sin(theta),
                0.0,
            ],
        ],
        dtype=float,
    )
    positions: list[tuple[float, float, float]] = []
    for layer, oxygen_z in enumerate(WATER_LAYER_Z):
        offset = 0.5 * layer
        for index_x in range(grid_x):
            for index_y in range(grid_y):
                oxygen = np.asarray(
                    [
                        ((index_x + 0.5 + offset) * step_x) % length_x,
                        ((index_y + 0.5 + offset) * step_y) % length_y,
                        oxygen_z,
                    ],
                    dtype=float,
                )
                quarter_turn = (index_x + 2 * index_y + layer) % 4
                angle = quarter_turn * np.pi / 2.0
                rotation = np.asarray(
                    [
                        [np.cos(angle), -np.sin(angle), 0.0],
                        [np.sin(angle), np.cos(angle), 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                )
                water = reference @ rotation.T + oxygen
                water[:, 0] %= length_x
                water[:, 1] %= length_y
                positions.extend(tuple(position) for position in water)
    return positions


def _build_large_example():
    from ase import Atoms

    repeated = _read_verified_base_structure().repeat(SUPERCELL)
    water_positions = _water_positions(repeated.cell.lengths())
    water_count = len(WATER_LAYER_Z) * WATER_GRID[0] * WATER_GRID[1]
    waters = Atoms(
        symbols="OHH" * water_count,
        positions=water_positions,
        cell=repeated.cell,
        pbc=repeated.pbc,
    )
    atoms = repeated + waters
    atoms.info.update(
        {
            "meia_example_source": "examples/CONTCAR",
            "meia_example_source_sha256": BASE_STRUCTURE_SHA256,
            "meia_example_transformation": "repeat_5x5x1_add_200_H2O",
        }
    )
    if len(atoms) != 6225:
        raise RuntimeError(f"large example atom count changed: {len(atoms)}")
    return atoms


def _payloads() -> dict[Path, bytes]:
    sys.path.insert(0, str(PROJECT_ROOT))

    from ase.io import write

    from meia import __version__
    from meia.atom_styles import emphasize_subject, replace_selected_indices
    from meia.periodic_display import CellPeriodicSettings, PeriodicRange
    from meia.presets import (
        PresetKind,
        PresetMetadata,
        SCHEMA_VERSION,
        SnapshotStructure,
        WorkspaceSnapshot,
        load_default_style,
        workspace_snapshot_to_json,
    )
    from meia.visual_state import (
        VisualizationState,
        merge_portable_style_for_structure,
    )
    atoms = _build_large_example()
    structure_buffer = io.StringIO()
    write(structure_buffer, atoms, format="extxyz")
    structure_payload = structure_buffer.getvalue().encode("utf-8")

    default = load_default_style()
    style = merge_portable_style_for_structure(default.style, atoms)
    style = replace(
        style,
        cell_periodic=CellPeriodicSettings(
            show_unit_cell=2,
            unwrap_bonded_groups=True,
            a=PeriodicRange(0, 1),
            b=PeriodicRange(0, 1),
            c=PeriodicRange(0, 1),
        ),
    )
    first_water = 225 * SUPERCELL[0] * SUPERCELL[1] * SUPERCELL[2]
    selection = replace_selected_indices(
        VisualizationState(style=style).atom_selection,
        (first_water, first_water + 1, first_water + 2),
        len(atoms),
    )
    selection = emphasize_subject(atoms, selection, 0.30)
    state = VisualizationState(style=style, atom_selection=selection)
    snapshot = WorkspaceSnapshot(
        metadata=PresetMetadata(
            schema_version=SCHEMA_VERSION,
            preset_kind=PresetKind.WORKSPACE_SNAPSHOT,
            name="MEIA CONTCAR 5x5 supercell with 200 added water molecules",
            created_at=CREATED_AT,
            meia_version=__version__,
        ),
        structure=SnapshotStructure.from_atoms(atoms, STRUCTURE_NAME),
        state=state,
    )
    snapshot_payload = workspace_snapshot_to_json(snapshot).encode("utf-8")
    return {
        EXAMPLES_DIR / STRUCTURE_NAME: structure_payload,
        EXAMPLES_DIR / SNAPSHOT_NAME: snapshot_payload,
    }


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o644)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate or verify MEIA's synthetic large-system example."
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payloads = _payloads()
    if args.check:
        mismatches = [
            path.name
            for path, payload in payloads.items()
            if not path.is_file() or path.read_bytes() != payload
        ]
        if mismatches:
            parser.error("large example is missing or stale: " + ", ".join(mismatches))
        print("large-system example is current")
        return 0

    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for path, payload in payloads.items():
        _atomic_write(path, payload)
        print(f"wrote {path.relative_to(PROJECT_ROOT)} ({len(payload)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
