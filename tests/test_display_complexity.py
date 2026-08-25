"""大体系显示复杂度与自动预览策略。"""

import numpy as np
from ase import Atoms

from meia.display_complexity import DisplayComplexity, measure_display_complexity
from meia.visual_state import VisualizationState, resolve_render_context


def test_display_complexity_uses_render_artist_and_instance_thresholds():
    ordinary = DisplayComplexity.from_counts(900, 900, 500, 200)
    assert ordinary.estimated_2d_artist_count == 4_100
    assert ordinary.manual_2d_recommended is False
    assert ordinary.large_3d_interaction is False

    large_2d = DisplayComplexity.from_counts(900, 900, 650, 200)
    assert large_2d.estimated_2d_artist_count == 5_000
    assert large_2d.manual_2d_recommended is True

    large_3d = DisplayComplexity.from_counts(1_000, 5_000, 0, 0)
    assert large_3d.large_3d_interaction is True
    assert large_3d.extreme_3d_interaction is False

    extreme = DisplayComplexity.from_counts(1_000, 20_000, 0, 0)
    assert extreme.extreme_3d_interaction is True


def test_measure_display_complexity_counts_real_visible_context_without_mutation():
    atoms = Atoms(
        "OHH",
        positions=[[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
    )
    snapshot = atoms.positions.copy()
    context = resolve_render_context(atoms, VisualizationState())

    measured = measure_display_complexity(len(atoms), context)

    assert measured.source_atom_count == 3
    assert measured.atom_instance_count == 3
    assert measured.visible_bond_instance_count == 2
    assert measured.hydrogen_bond_instance_count == 0
    assert measured.estimated_2d_artist_count == 15
    assert np.array_equal(atoms.positions, snapshot)
