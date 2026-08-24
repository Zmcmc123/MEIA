"""
化学键 2D 几何计算模块。

椭圆几何：
- 长轴（垂直于键方向）= 键宽（bond_width），始终与矩形短边等长且外接
- 短轴（沿键方向）= 穹顶深度，可随原子半径缩放
"""

from dataclasses import dataclass
from typing import List, Sequence
import numpy as np

from .bond_rules import normalize_element_pair
from .bond_segments import clip_bond_to_spheres
from .atom_styles import apply_color_strength
from .periodic_display import BondDisplayInstance, LatticeShift
from .projection import ProjectionResult
from .config import DEFAULT_STROKE_COLOR, RenderConfig


@dataclass
class BondGeometry:
    """一根键的 2D 几何参数。"""
    pos_a: np.ndarray
    pos_b: np.ndarray

    # 矩形（无描边，纯填充）
    rect_a_corners: np.ndarray
    rect_a_color: str
    rect_b_corners: np.ndarray
    rect_b_color: str

    # 椭圆（有描边，在矩形下层）
    # Matplotlib Ellipse: width=长轴, height=短轴, angle 使长轴垂直于键方向
    ellipse_a_center: np.ndarray
    ellipse_a_width: float    # 长轴 = bond_width（恒定）
    ellipse_a_height: float   # 短轴 = 2 * dome_depth
    ellipse_a_angle: float
    ellipse_a_color: str

    ellipse_b_center: np.ndarray
    ellipse_b_width: float
    ellipse_b_height: float
    ellipse_b_angle: float
    ellipse_b_color: str

    # 平行线（外侧粗描边，沿键方向两侧）
    line_1_start: np.ndarray
    line_1_end: np.ndarray
    line_2_start: np.ndarray
    line_2_end: np.ndarray

    depth: float       # 兼容字段：整根键的中点深度
    depth_a: float     # A 端原子的投影深度
    depth_b: float     # B 端原子的投影深度
    group_depth: float
    bond_angle: float

    # SVG 编组与 Illustrator 可追溯元数据
    bond_id: str
    atom_i: int
    atom_j: int
    atom_i_image_shift: LatticeShift
    atom_j_image_shift: LatticeShift
    element_pair: tuple[str, str]
    periodic_offset: tuple[int, int, int]
    stroke_color: str = DEFAULT_STROKE_COLOR


def compute_bond_geometries(
    bond_instances: Sequence[BondDisplayInstance],
    proj: ProjectionResult,
    config: RenderConfig,
) -> List[BondGeometry]:
    geometries = []

    if proj.natoms == 0 or not bond_instances:
        return geometries

    mean_radius = proj.radii_2d.mean()
    bond_width = mean_radius * config.bond_width_ratio
    dome_depth = bond_width * config.ellipse_ry_ratio
    half_width = bond_width / 2

    for bond_instance in bond_instances:
        bond = bond_instance.source_bond
        if not bond.visible:
            continue
        atom_i_row = proj.instance_index_by_key.get(bond_instance.atom_i_key)
        atom_j_row = proj.instance_index_by_key.get(bond_instance.atom_j_key)
        if atom_i_row is None or atom_j_row is None:
            continue
        center_a = np.array([
            proj.positions_2d[atom_i_row, 0],
            proj.positions_2d[atom_i_row, 1],
            proj.depths[atom_i_row],
        ])
        center_b = np.array([
            proj.positions_2d[atom_j_row, 0],
            proj.positions_2d[atom_j_row, 1],
            proj.depths[atom_j_row],
        ])

        segment = clip_bond_to_spheres(
            center_a,
            center_b,
            proj.radii_2d[atom_i_row],
            proj.radii_2d[atom_j_row],
        )
        if segment is None:
            continue

        pos_a = segment.start[:2]
        pos_b = segment.end[:2]
        color_a = proj.colors[atom_i_row]
        color_b = proj.colors[atom_j_row]
        if proj.color_strengths is None:
            bond_strength = 1.0
        else:
            bond_strength = min(
                float(proj.color_strengths[atom_i_row]),
                float(proj.color_strengths[atom_j_row]),
            )
        stroke_color = apply_color_strength(
            config.effective_bond_stroke_color,
            bond_strength,
        )

        delta = pos_b - pos_a
        dist_2d = np.linalg.norm(delta)
        if dist_2d < 1e-6:
            continue

        direction = delta / dist_2d
        normal = np.array([-direction[1], direction[0]])

        bond_angle = np.arctan2(delta[1], delta[0])
        bond_angle_deg = np.degrees(bond_angle)
        ellipse_angle = bond_angle_deg + 90.0

        midpoint = segment.midpoint[:2]

        # 两端椭圆帽与键宽保持固定比例，不再按端点原子半径分别缩放。
        ew_a = ew_b = bond_width
        eh_a = eh_b = 2 * dome_depth

        # 矩形四角
        rect_a_corners = np.array([
            pos_a + normal * half_width,
            pos_a - normal * half_width,
            midpoint - normal * half_width,
            midpoint + normal * half_width,
        ])
        rect_b_corners = np.array([
            midpoint + normal * half_width,
            midpoint - normal * half_width,
            pos_b - normal * half_width,
            pos_b + normal * half_width,
        ])

        # 平行线
        line_1_start = pos_a + normal * half_width
        line_1_end = pos_b + normal * half_width
        line_2_start = pos_a - normal * half_width
        line_2_end = pos_b - normal * half_width

        depth_a = float(segment.start[2])
        depth_b = float(segment.end[2])
        group_depth = float(segment.midpoint[2])
        element_pair = normalize_element_pair(
            proj.symbols[atom_i_row],
            proj.symbols[atom_j_row],
        )
        periodic_offset = tuple(int(value) for value in bond.offset)
        atom_i_image_shift = tuple(
            int(value) for value in proj.image_shifts[atom_i_row]
        )
        atom_j_image_shift = tuple(
            int(value) for value in proj.image_shifts[atom_j_row]
        )

        geom = BondGeometry(
            pos_a=pos_a, pos_b=pos_b,
            rect_a_corners=rect_a_corners, rect_a_color=color_a,
            rect_b_corners=rect_b_corners, rect_b_color=color_b,
            ellipse_a_center=pos_a,
            ellipse_a_width=ew_a, ellipse_a_height=eh_a,
            ellipse_a_angle=ellipse_angle, ellipse_a_color=color_a,
            ellipse_b_center=pos_b,
            ellipse_b_width=ew_b, ellipse_b_height=eh_b,
            ellipse_b_angle=ellipse_angle, ellipse_b_color=color_b,
            line_1_start=line_1_start, line_1_end=line_1_end,
            line_2_start=line_2_start, line_2_end=line_2_end,
            stroke_color=stroke_color,
            depth=group_depth, depth_a=depth_a, depth_b=depth_b,
            group_depth=group_depth,
            bond_angle=bond_angle,
            bond_id=bond_instance.bond_instance_id,
            atom_i=bond.i,
            atom_j=bond.j,
            atom_i_image_shift=atom_i_image_shift,
            atom_j_image_shift=atom_j_image_shift,
            element_pair=element_pair,
            periodic_offset=periodic_offset,
        )
        geometries.append(geom)

    return geometries
