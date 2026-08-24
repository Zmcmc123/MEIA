"""化学键与原子的深度遮挡回归测试。"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Polygon
import numpy as np
from types import MappingProxyType

from meia.bond_rules import ResolvedBond
from meia.config import RenderConfig
from meia.geometry import compute_bond_geometries
from meia.periodic_display import BondDisplayInstance
from meia.projection import ProjectionResult
from meia.renderer import render


ZERO_SHIFT = (0, 0, 0)


def _slanted_bond_instance() -> BondDisplayInstance:
    source = ResolvedBond(
        i=0,
        j=1,
        offset=ZERO_SHIFT,
        distance=5.0,
        pair=("C", "O"),
        bond_id="bond_source",
        visible=True,
        visibility_source="pair_enabled",
    )
    return BondDisplayInstance(
        source_bond=source,
        atom_i_key=(0, ZERO_SHIFT),
        atom_j_key=(1, ZERO_SHIFT),
        bond_instance_id="bond_instance",
    )


def _slanted_bond_projection() -> ProjectionResult:
    """两个投影端点重合于键轴，但三维深度一远一近。"""
    return ProjectionResult(
        positions_2d=np.array([[0.0, 1.0], [4.0, 1.0]]),
        depths=np.array([0.0, 2.0]),
        radii_2d=np.array([1.0, 1.0]),
        colors=["#3F4F6A", "#E5A6A6"],
        symbols=["C", "O"],
        numbers=np.array([6, 8]),
        scale=1.0,
        width=4.0,
        height=2.0,
        rotation_matrix=np.eye(3),
        cell_vertices_2d=None,
        cell_lines_2d=None,
        source_atom_indices=np.array([0, 1]),
        instance_keys=((0, ZERO_SHIFT), (1, ZERO_SHIFT)),
        image_shifts=np.array([[0, 0, 0], [0, 0, 0]]),
        instance_index_by_key=MappingProxyType(
            {(0, ZERO_SHIFT): 0, (1, ZERO_SHIFT): 1}
        ),
    )


def test_whole_bond_uses_one_midpoint_depth_between_endpoint_atoms():
    """整根键共享中点深度，并整体位于远、近两个端点原子之间。"""
    proj = _slanted_bond_projection()
    config = RenderConfig(show_unit_cell=0)
    geometries = compute_bond_geometries([_slanted_bond_instance()], proj, config)

    fig = render(proj, geometries, config)
    ax = fig.axes[0]

    atoms = sorted(
        (patch for patch in ax.patches if isinstance(patch, Circle)),
        key=lambda patch: patch.center[0],
    )
    bond_rectangles = sorted(
        (patch for patch in ax.patches if isinstance(patch, Polygon)),
        key=lambda patch: patch.get_xy()[:, 0].mean(),
    )

    far_atom_z = atoms[0].get_zorder()
    near_atom_z = atoms[1].get_zorder()
    rectangle_zorders = {patch.get_zorder() for patch in bond_rectangles}

    assert len(rectangle_zorders) == 1
    rectangle_z = rectangle_zorders.pop()
    assert far_atom_z < rectangle_z < near_atom_z
    plt.close(fig)


def test_bond_caps_share_whole_bond_depth():
    """两端椭圆帽共享整根键的中点深度。"""
    proj = _slanted_bond_projection()
    config = RenderConfig(show_unit_cell=0)
    geometries = compute_bond_geometries([_slanted_bond_instance()], proj, config)

    fig = render(proj, geometries, config)
    ax = fig.axes[0]

    atoms = sorted(
        (patch for patch in ax.patches if isinstance(patch, Circle)),
        key=lambda patch: patch.center[0],
    )
    caps = sorted(
        (patch for patch in ax.patches if type(patch) is Ellipse),
        key=lambda patch: patch.center[0],
    )

    assert caps[0].get_zorder() == caps[1].get_zorder()
    assert atoms[0].get_zorder() < caps[0].get_zorder() < atoms[1].get_zorder()
    plt.close(fig)


def test_each_bond_cap_is_below_its_matching_rectangle():
    """无论键的深度方向如何，两端椭圆帽都只能露出矩形外侧的半圈。"""
    proj = _slanted_bond_projection()
    config = RenderConfig(show_unit_cell=0)
    geometries = compute_bond_geometries([_slanted_bond_instance()], proj, config)

    fig = render(proj, geometries, config)
    ax = fig.axes[0]
    bond_children = [
        artist
        for artist in ax.get_children()
        if getattr(artist, "_meia_bond_id", None)
    ]
    roles = [artist._meia_bond_role for artist in bond_children]

    assert roles == [
        "cap-a",
        "cap-b",
        "rect-a",
        "rect-b",
        "outline-1",
        "outline-2",
    ]
    assert len({artist.get_zorder() for artist in bond_children}) == 1
    plt.close(fig)


def test_bond_outline_uses_two_complete_lines_at_whole_bond_depth():
    """整键排序后两侧描边各保持一条完整线，不再从中点拆分。"""
    proj = _slanted_bond_projection()
    config = RenderConfig(show_unit_cell=0)
    geometries = compute_bond_geometries([_slanted_bond_instance()], proj, config)

    fig = render(proj, geometries, config)
    ax = fig.axes[0]

    atoms = sorted(
        (patch for patch in ax.patches if isinstance(patch, Circle)),
        key=lambda patch: patch.center[0],
    )
    lines = list(ax.lines)

    assert len(lines) == 2
    assert all(len(line.get_xdata()) == 2 for line in lines)
    assert len({line.get_zorder() for line in lines}) == 1
    assert atoms[0].get_zorder() < lines[0].get_zorder() < atoms[1].get_zorder()
    assert all(line.get_linewidth() == config.bond_stroke_width for line in lines)
    plt.close(fig)
