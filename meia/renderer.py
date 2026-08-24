"""
渲染管线模块。

Z-order 策略（v8）：

  原子 zorder = depth × 100000 + 4
  整键 zorder = midpoint_depth × 100000

  乘数 100000 确保 depth 差异（0~10）产生的 zorder 差距（0~1000000）
  远大于 sub_order 范围（1~4），保证不同 depth 的元素严格按深度排序。
  整键的六个对象使用相同 zorder，并按插入顺序决定组内绘制次序。

  每根键先在三维空间裁剪到两个显示球的球面交点，并以裁剪后键段
  中点深度作为整根键的全局深度。键的六个可见对象共享该深度，
  再由组内插入顺序保证椭圆帽始终位于矩形下方。

  键内部：椭圆(有描边) → 矩形(无描边) → 平行线(纯描边路径)
  椭圆在矩形下层 → 内侧被遮挡 → 只露外侧弧线
  平行线为纯描边线条，无填充，线宽略细于原子轮廓。
"""

from typing import List, Optional, Sequence
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Ellipse
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from .projection import ProjectionResult
from .geometry import BondGeometry
from .config import RenderConfig
from .hydrogen_bonds import (
    HYDROGEN_BOND_2D_WIDTH,
    HydrogenBondGeometry,
)

_DEPTH_SCALE = 100000


def render(
    proj: ProjectionResult,
    bond_geoms: List[BondGeometry],
    config: RenderConfig,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
    hydrogen_bond_geoms: Sequence[HydrogenBondGeometry] = (),
) -> plt.Figure:
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=config.dpi)

    ax.set_aspect("equal")
    ax.set_axis_off()

    items = []
    for i in range(proj.natoms):
        z = proj.depths[i] * _DEPTH_SCALE + 4
        items.append((z, "atom", i))
    for k, geom in enumerate(bond_geoms):
        items.append((geom.group_depth * _DEPTH_SCALE, "bond", k))
    for k, geom in enumerate(hydrogen_bond_geoms):
        items.append((geom.group_depth * _DEPTH_SCALE, "hydrogen_bond", k))

    items.sort(key=lambda x: x[0])

    for z, item_type, idx in items:
        if item_type == "atom":
            _draw_atom(ax, proj, idx, config, z)
        elif item_type == "bond":
            _draw_bond(ax, bond_geoms[idx], config, z)
        elif item_type == "hydrogen_bond":
            _draw_hydrogen_bond(ax, hydrogen_bond_geoms[idx], z)

    if proj.cell_lines_2d is not None:
        _draw_cell_lines(ax, proj)

    fig._meia_bond_manifest = {
        geom.bond_id: _bond_metadata(geom)
        for geom in bond_geoms
    }
    fig._meia_atom_manifest = {
        _atom_instance_id(proj, index): _atom_metadata(proj, index)
        for index in range(proj.natoms)
    }
    fig._meia_hydrogen_bond_manifest = {
        geom.hydrogen_bond.instance_id: _hydrogen_bond_metadata(geom)
        for geom in hydrogen_bond_geoms
    }
    ax.set_xlim(0, proj.width)
    ax.set_ylim(0, proj.height)
    return fig


def _draw_atom(ax, proj, i, config, zorder):
    xy = proj.positions_2d[i]
    r = proj.radii_2d[i]
    circle = Circle(
        (xy[0], xy[1]), r,
        facecolor=proj.colors[i],
        edgecolor=(
            proj.outline_colors[i]
            if proj.outline_colors is not None
            else config.stroke_color
        ),
        linewidth=config.outline_width,
        clip_on=False,
    )
    circle.set_gid(_atom_instance_id(proj, i))
    circle.set_zorder(zorder)
    ax.add_patch(circle)


def _atom_instance_id(proj: ProjectionResult, index: int) -> str:
    source_atom_index, replica_translation = proj.instance_keys[index]
    return "atom_{}_{}_{}_{}".format(source_atom_index, *replica_translation)


def _atom_metadata(proj: ProjectionResult, index: int) -> dict:
    source_atom_index, replica_translation = proj.instance_keys[index]
    return {
        "source_atom_index": int(source_atom_index),
        "replica_translation": tuple(int(value) for value in replica_translation),
        "image_shift": tuple(int(value) for value in proj.image_shifts[index]),
    }


def _bond_metadata(geom: BondGeometry) -> dict:
    return {
        "bond_id": geom.bond_id,
        "atom_i": geom.atom_i,
        "atom_j": geom.atom_j,
        "atom_i_image_shift": geom.atom_i_image_shift,
        "atom_j_image_shift": geom.atom_j_image_shift,
        "elements": geom.element_pair,
        "periodic_offset": geom.periodic_offset,
    }


def _hydrogen_bond_metadata(geom: HydrogenBondGeometry) -> dict:
    candidate = geom.hydrogen_bond.candidate
    return {
        "donor_oxygen": candidate.donor_oxygen,
        "hydrogen": candidate.hydrogen,
        "acceptor_oxygen": candidate.acceptor_oxygen,
        "donor_oxygen_image_shift": geom.donor_oxygen_image_shift,
        "hydrogen_image_shift": geom.hydrogen_image_shift,
        "acceptor_oxygen_image_shift": geom.acceptor_oxygen_image_shift,
        "donor_oxygen_offset_from_hydrogen": (
            candidate.donor_oxygen_offset_from_hydrogen
        ),
        "acceptor_offset_from_hydrogen": candidate.acceptor_offset_from_hydrogen,
        "hydrogen_acceptor_distance": candidate.hydrogen_acceptor_distance,
        "angle_degrees": candidate.angle_degrees,
        "color_strength": geom.hydrogen_bond.color_strength,
        "visibility_source": geom.hydrogen_bond.visibility_source,
    }


def _tag_bond_artist(artist, geom: BondGeometry, role: str):
    artist.set_gid(f"{geom.bond_id}__{role}")
    artist._meia_bond_id = geom.bond_id
    artist._meia_bond_role = role
    artist._meia_bond_metadata = _bond_metadata(geom)
    return artist


def _draw_bond(ax, geom: BondGeometry, config: RenderConfig, zorder: float):
    """按固定内部顺序绘制一根整键的六个可见对象。"""
    cap_z = rect_z = line_z = zorder
    _tag_bond_artist(
        _draw_ellipse(
            ax,
            geom.ellipse_a_center,
            geom.ellipse_a_width,
            geom.ellipse_a_height,
            geom.ellipse_a_angle,
            geom.ellipse_a_color,
            geom.stroke_color,
            config,
            cap_z,
        ),
        geom,
        "cap-a",
    )
    _tag_bond_artist(
        _draw_ellipse(
            ax,
            geom.ellipse_b_center,
            geom.ellipse_b_width,
            geom.ellipse_b_height,
            geom.ellipse_b_angle,
            geom.ellipse_b_color,
            geom.stroke_color,
            config,
            cap_z,
        ),
        geom,
        "cap-b",
    )
    _tag_bond_artist(
        _draw_rect(ax, geom.rect_a_corners, geom.rect_a_color, rect_z),
        geom,
        "rect-a",
    )
    _tag_bond_artist(
        _draw_rect(ax, geom.rect_b_corners, geom.rect_b_color, rect_z),
        geom,
        "rect-b",
    )
    for role, line in zip(
        ("outline-1", "outline-2"),
        _draw_bond_lines(ax, geom, config, line_z),
    ):
        _tag_bond_artist(line, geom, role)


def _draw_ellipse(
    ax,
    center,
    width,
    height,
    angle,
    color,
    stroke_color,
    config,
    zorder,
):
    """绘制一端带描边的椭圆帽。"""
    ell = Ellipse(
        xy=center, width=width, height=height, angle=angle,
        facecolor=color, edgecolor=stroke_color,
        linewidth=config.bond_stroke_width,
        clip_on=False,
    )
    ell.set_zorder(zorder)
    ax.add_patch(ell)
    return ell


def _draw_rect(ax, corners, color, zorder):
    """绘制一段无描边的纯填充键矩形。"""
    rect = Polygon(
        corners, closed=True,
        facecolor=color, edgecolor="none", linewidth=0,
        clip_on=False,
    )
    rect.set_zorder(zorder)
    ax.add_patch(rect)
    return rect


def _draw_bond_lines(ax, geom, config, zorder):
    """绘制整根键的两条完整平行描边线。"""
    stroke = geom.stroke_color
    lw = config.bond_stroke_width

    lines = []
    for start, end in [
        (geom.line_1_start, geom.line_1_end),
        (geom.line_2_start, geom.line_2_end),
    ]:
        line = Line2D(
            [start[0], end[0]],
            [start[1], end[1]],
            color=stroke,
            linewidth=lw,
            solid_capstyle="butt",
            fillstyle="none",
            markeredgewidth=0,
            clip_on=False,
        )
        line.set_zorder(zorder)
        ax.add_line(line)
        lines.append(line)
    return lines


def _draw_hydrogen_bond(
    ax,
    geometry: HydrogenBondGeometry,
    zorder: float,
) -> Line2D:
    """绘制从 H 表面到受体 O 表面的可追溯虚线。"""
    line = Line2D(
        [geometry.start[0], geometry.end[0]],
        [geometry.start[1], geometry.end[1]],
        color=geometry.color,
        linewidth=HYDROGEN_BOND_2D_WIDTH,
        linestyle="--",
        dash_capstyle="round",
        clip_on=False,
    )
    line.set_gid(geometry.hydrogen_bond.instance_id)
    line.set_zorder(zorder)
    ax.add_line(line)
    return line


def _draw_cell_lines(ax, proj):
    cell_data = proj.cell_lines_2d
    if cell_data is None:
        return
    positions = cell_data["positions"]
    T = cell_data["T"]
    D = cell_data["D"]
    if D is None:
        return
    for a in range(len(positions)):
        c = T[a]
        if c == -1:
            continue
        xy = positions[a]
        hxy = D[c]
        patch = PathPatch(
            Path((xy + hxy, xy - hxy)),
            facecolor="none", edgecolor="gray",
            linewidth=0.5, linestyle="--",
            clip_on=False,
        )
        ax.add_patch(patch)
