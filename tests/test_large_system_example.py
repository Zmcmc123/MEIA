"""Public synthetic large-system example and workspace snapshot."""

from pathlib import Path
import subprocess
import sys

from meia.display_complexity import measure_display_complexity
from meia.io import read_structure
from meia.presets import WorkspaceSnapshot, parse_preset
from meia.render_topology import build_render_topology, compose_render_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATOR = PROJECT_ROOT / "scripts" / "generate_large_system_example.py"
STRUCTURE = PROJECT_ROOT / "examples" / "MEIA_large_slab_water_6246.xyz"
SNAPSHOT = (
    PROJECT_ROOT
    / "examples"
    / "MEIA_large_slab_water_6246.workspace.meia.json"
)


def test_large_system_example_is_reproducible_and_exercises_extreme_3d():
    result = subprocess.run(
        [sys.executable, str(GENERATOR), "--check"],
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    atoms = read_structure(str(STRUCTURE))
    snapshot = parse_preset(SNAPSHOT.read_bytes())
    assert isinstance(snapshot, WorkspaceSnapshot)
    assert len(atoms) == 6246
    assert len(snapshot.structure.symbols) == len(atoms)
    assert snapshot.structure.source_name == STRUCTURE.name
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
    assert complexity.atom_instance_count == 24_984
    assert complexity.extreme_3d_interaction is True
