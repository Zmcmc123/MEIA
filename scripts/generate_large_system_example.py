#!/usr/bin/env python3
"""Generate the public synthetic large slab-and-water test workspace."""

from __future__ import annotations

import argparse
from dataclasses import replace
import io
from pathlib import Path
import os
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = PROJECT_ROOT / "examples"
STRUCTURE_NAME = "MEIA_large_slab_water_6246.xyz"
SNAPSHOT_NAME = "MEIA_large_slab_water_6246.workspace.meia.json"
CREATED_AT = "2026-08-25T00:00:00+08:00"
NX = 32
WATER_LAYERS = 2


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
    from scripts.benchmark_large_system import generate_slab_water_system

    atoms = generate_slab_water_system(NX, WATER_LAYERS)
    if len(atoms) != 6246:
        raise RuntimeError(f"large example atom count changed: {len(atoms)}")
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
            a=PeriodicRange(0, 2),
            b=PeriodicRange(0, 2),
            c=PeriodicRange(0, 1),
        ),
    )
    first_water = 3 * NX * NX
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
            name="MEIA large slab and water interaction test",
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
