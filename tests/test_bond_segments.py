"""三维球面裁剪后的化学键段测试。"""

import numpy as np
import pytest
from types import MappingProxyType

from meia.bond_segments import clip_bond_to_spheres
from meia.bond_rules import ResolvedBond
from meia.config import RenderConfig
from meia.geometry import compute_bond_geometries
from meia.periodic_display import BondDisplayInstance
from meia.projection import ProjectionResult


ZERO_SHIFT = (0, 0, 0)


def _bond_instance(*, visible=True, atom_j_replica=ZERO_SHIFT):
    source = ResolvedBond(
        i=0,
        j=1,
        offset=ZERO_SHIFT,
        distance=5.0,
        pair=("C", "O"),
        bond_id="bond_source",
        visible=visible,
        visibility_source="pair_enabled" if visible else "pair_disabled",
    )
    return BondDisplayInstance(
        source_bond=source,
        atom_i_key=(0, ZERO_SHIFT),
        atom_j_key=(1, atom_j_replica),
        bond_instance_id="bond_instance",
    )


def test_clip_bond_to_unequal_spheres_uses_surface_intersections():
    """若仍返回原子中心，两个手算球面交点断言必须失败。"""
    segment = clip_bond_to_spheres(
        np.array([0.0, 0.0, 0.0]),
        np.array([10.0, 0.0, 0.0]),
        radius_a=2.0,
        radius_b=1.0,
    )

    assert segment is not None
    assert np.allclose(segment.start, [2.0, 0.0, 0.0])
    assert np.allclose(segment.end, [9.0, 0.0, 0.0])
    assert np.allclose(segment.midpoint, [5.5, 0.0, 0.0])


def test_clip_bond_to_spheres_preserves_direction_for_oblique_bond():
    """三维裁剪不得退化为只在二维投影平面求圆交点。"""
    segment = clip_bond_to_spheres(
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 4.0, 0.0]),
        radius_a=1.0,
        radius_b=2.0,
    )

    assert segment is not None
    assert np.allclose(segment.start, [0.6, 0.8, 0.0])
    assert np.allclose(segment.end, [1.8, 2.4, 0.0])


def test_clip_bond_to_overlapping_spheres_has_no_visible_segment():
    """两个显示球相交时不得生成方向反转的外部键段。"""
    assert clip_bond_to_spheres(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        radius_a=0.6,
        radius_b=0.5,
    ) is None


def test_2d_geometry_projects_3d_surface_endpoints_and_fixed_caps():
    """端点深度必须来自球面交点，帽尺寸只随键宽而非原子半径变化。"""
    projection = ProjectionResult(
        positions_2d=np.array([[0.0, 1.0], [4.0, 1.0]]),
        depths=np.array([0.0, 3.0]),
        radii_2d=np.array([1.0, 0.5]),
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
    config = RenderConfig(
        bond_width_ratio=0.6,
        ellipse_ry_ratio=0.30,
        show_unit_cell=0,
    )

    geometries = compute_bond_geometries([_bond_instance()], projection, config)

    assert len(geometries) == 1
    geometry = geometries[0]
    assert np.allclose(geometry.pos_a, [0.8, 1.0])
    assert np.allclose(geometry.pos_b, [3.6, 1.0])
    assert geometry.depth_a == 0.6
    assert geometry.depth_b == 2.7
    assert np.allclose(geometry.ellipse_a_center, geometry.pos_a)
    assert np.allclose(geometry.ellipse_b_center, geometry.pos_b)
    assert geometry.ellipse_a_width == geometry.ellipse_b_width == pytest.approx(0.45)
    assert geometry.ellipse_a_height == geometry.ellipse_b_height == pytest.approx(0.27)
    assert geometry.bond_id == "bond_instance"
    assert geometry.atom_i == 0
    assert geometry.atom_j == 1
    assert geometry.atom_i_image_shift == ZERO_SHIFT
    assert geometry.atom_j_image_shift == ZERO_SHIFT


def test_2d_geometry_filters_invisible_or_missing_periodic_instances():
    """不可见源键或缺少任一显示端点时不得产生悬空几何。"""
    projection = ProjectionResult(
        positions_2d=np.array([[0.0, 1.0], [4.0, 1.0]]),
        depths=np.array([0.0, 0.0]),
        radii_2d=np.array([0.5, 0.5]),
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
    config = RenderConfig(show_unit_cell=0)

    assert compute_bond_geometries(
        [_bond_instance(visible=False)], projection, config
    ) == []
    assert compute_bond_geometries(
        [_bond_instance(atom_j_replica=(1, 0, 0))], projection, config
    ) == []
