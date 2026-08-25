#!/usr/bin/env python3
"""Deterministic slab-and-water benchmark for MEIA large-system paths."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_slab_water_system(nx: int, water_layers: int):
    """Return a deterministic three-layer square Si slab with water grids."""
    if isinstance(nx, bool) or not isinstance(nx, int) or nx < 1:
        raise ValueError("--nx must be at least 1")
    if (
        isinstance(water_layers, bool)
        or not isinstance(water_layers, int)
        or water_layers < 0
    ):
        raise ValueError("--water-layers must be at least 0")

    import numpy as np
    from ase import Atoms

    slab_spacing = 2.15
    cell_x = nx * slab_spacing
    cell_y = nx * slab_spacing
    symbols: list[str] = []
    positions: list[tuple[float, float, float]] = []
    for layer in range(3):
        z = 3.0 + layer * slab_spacing
        offset = 0.5 * slab_spacing if layer % 2 else 0.0
        for ix in range(nx):
            for iy in range(nx):
                symbols.append("Si")
                positions.append(
                    (
                        ((ix + 0.5) * slab_spacing + offset) % cell_x,
                        ((iy + 0.5) * slab_spacing + offset) % cell_y,
                        z,
                    )
                )

    water_spacing_target = 2.9
    water_nx = max(1, int(cell_x // water_spacing_target))
    water_ny = max(1, int(cell_y // water_spacing_target))
    water_dx = cell_x / water_nx
    water_dy = cell_y / water_ny
    water_base_z = 3.0 + 2 * slab_spacing + 3.0
    for layer in range(water_layers):
        z = water_base_z + layer * 3.0
        for ix in range(water_nx):
            for iy in range(water_ny):
                oxygen = np.asarray(
                    [
                        0.5 + ix * water_dx,
                        0.5 + iy * water_dy,
                        z,
                    ],
                    dtype=float,
                )
                symbols.extend(("O", "H", "H"))
                positions.extend(
                    (
                        tuple(oxygen),
                        tuple(oxygen + np.asarray([0.96, 0.0, 0.0])),
                        tuple(oxygen + np.asarray([-0.24, 0.93, 0.0])),
                    )
                )

    cell_z = water_base_z + max(water_layers - 1, 0) * 3.0 + 8.0
    return Atoms(
        symbols=symbols,
        positions=positions,
        cell=[cell_x, cell_y, cell_z],
        pbc=True,
    )


def _positive(parser: argparse.ArgumentParser, value: int, option: str) -> int:
    if value < 1:
        parser.error(f"{option} must be at least 1")
    return value


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _elapsed(callable_):
    started = perf_counter()
    value = callable_()
    return value, perf_counter() - started


def _run(args) -> dict[str, object]:
    import ase
    import matplotlib.pyplot as plt
    import numpy as np

    from meia import __version__
    from meia.display_complexity import measure_display_complexity
    from meia.i18n import I18n, Locale
    from meia.periodic_display import CellPeriodicSettings, PeriodicRange
    from meia.preview import render_preview_png
    from meia.render_topology import build_render_topology, compose_render_context
    from meia.view import render_2d
    from meia.viewer import create_3d_figure
    from meia.visual_state import PortableStyle, VisualizationState

    atoms = generate_slab_water_system(args.nx, args.water_layers)
    periodic = CellPeriodicSettings(
        show_unit_cell=2,
        a=PeriodicRange(0, args.repeat_a),
        b=PeriodicRange(0, args.repeat_b),
        c=PeriodicRange(0, args.repeat_c),
    )
    state = VisualizationState(
        style=PortableStyle(cell_periodic=periodic),
    )
    topology, topology_seconds = _elapsed(
        lambda: build_render_topology(
            atoms,
            state,
            structure_id="meia-large-system-benchmark",
        )
    )
    context, context_seconds = _elapsed(
        lambda: compose_render_context(
            atoms,
            state,
            topology,
            structure_id="meia-large-system-benchmark",
        )
    )
    figure3d, figure3d_seconds = _elapsed(
        lambda: create_3d_figure(
            atoms,
            context.config,
            render_context=context,
            figure_messages=I18n(Locale.EN).bundle("figure3d"),
        )
    )
    figure3d_json, json_seconds = _elapsed(figure3d.to_json)
    complexity = measure_display_complexity(len(atoms), context)

    render2d_seconds = None
    preview_seconds = None
    preview_png_bytes = None
    if not args.skip_2d:
        figure2d = None
        try:
            figure2d, render2d_seconds = _elapsed(
                lambda: render_2d(
                    atoms,
                    context.config,
                    render_context=context,
                )
            )
            preview_png, preview_seconds = _elapsed(
                lambda: render_preview_png(
                    figure2d,
                    transparent=context.config.transparent,
                )
            )
            preview_png_bytes = len(preview_png)
        finally:
            if figure2d is not None:
                plt.close(figure2d)

    return {
        "parameters": {
            "nx": args.nx,
            "water_layers": args.water_layers,
            "repeat_a": args.repeat_a,
            "repeat_b": args.repeat_b,
            "repeat_c": args.repeat_c,
            "skip_2d": args.skip_2d,
        },
        "source_atoms": len(atoms),
        "atom_instances": complexity.atom_instance_count,
        "visible_bond_instances": complexity.visible_bond_instance_count,
        "hydrogen_bond_instances": complexity.hydrogen_bond_instance_count,
        "estimated_2d_artists": complexity.estimated_2d_artist_count,
        "manual_2d_recommended": complexity.manual_2d_recommended,
        "large_3d_interaction": complexity.large_3d_interaction,
        "extreme_3d_interaction": complexity.extreme_3d_interaction,
        "timings_s": {
            "topology": topology_seconds,
            "context": context_seconds,
            "figure3d": figure3d_seconds,
            "figure3d_json": json_seconds,
            "render2d": render2d_seconds,
            "preview_png": preview_seconds,
        },
        "figure3d_json_bytes": len(figure3d_json.encode("utf-8")),
        "preview_png_bytes": preview_png_bytes,
        "peak_rss_bytes": _peak_rss_bytes(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "meia": __version__,
            "ase": ase.__version__,
            "numpy": np.__version__,
        },
    }


def _human_report(payload: dict[str, object]) -> str:
    timings = payload["timings_s"]
    assert isinstance(timings, dict)
    return "\n".join(
        (
            f"source atoms: {payload['source_atoms']}",
            f"display instances: {payload['atom_instances']}",
            f"visible bonds: {payload['visible_bond_instances']}",
            f"hydrogen bonds: {payload['hydrogen_bond_instances']}",
            f"topology: {timings['topology']:.6f} s",
            f"3D figure: {timings['figure3d']:.6f} s",
            f"3D JSON: {payload['figure3d_json_bytes']} bytes",
            f"manual 2D recommended: {payload['manual_2d_recommended']}",
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark deterministic MEIA large-system rendering paths."
    )
    parser.add_argument("--nx", type=int, default=10)
    parser.add_argument("--water-layers", type=int, default=2)
    parser.add_argument("--repeat-a", type=int, default=1)
    parser.add_argument("--repeat-b", type=int, default=1)
    parser.add_argument("--repeat-c", type=int, default=1)
    parser.add_argument("--skip-2d", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    _positive(parser, args.nx, "--nx")
    if args.water_layers < 0:
        parser.error("--water-layers must be at least 0")
    for option in ("repeat_a", "repeat_b", "repeat_c"):
        _positive(parser, getattr(args, option), "--" + option.replace("_", "-"))

    os.environ.setdefault(
        "MPLCONFIGDIR",
        str(Path(tempfile.gettempdir()) / "meia-benchmark-mpl-cache"),
    )
    sys.path.insert(0, str(PROJECT_ROOT))
    payload = _run(args)
    if args.json:
        print(json.dumps(payload, sort_keys=True, allow_nan=False))
    else:
        print(_human_report(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
