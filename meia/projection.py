"""
投影坐标系统。

借鉴 ASE PlottingVariables 的投影逻辑（旋转矩阵、坐标变换、缩放、偏移），
封装为独立函数，输出结构化的投影结果。

核心流程：
1. rotation 字符串 → 3×3 旋转矩阵
2. 原子 3D 坐标 × 旋转矩阵 → 旋转后坐标
3. 计算 bbox → 缩放 + 偏移 → 2D 屏幕坐标
4. Z 深度用于排序
"""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping
import numpy as np
from ase import Atoms
from ase.utils import rotate

from .config import RenderConfig
from .periodic_display import (
    AtomInstanceKey,
    CellPeriodicSettings,
    PeriodicDisplay,
    build_periodic_display,
)


@dataclass
class ProjectionResult:
    """投影结果。

    所有数组按可见显示实例排列；source_atom_indices 保留原始原子身份。
    """

    positions_2d: np.ndarray   # (N, 2)  2D 屏幕坐标
    depths: np.ndarray         # (N,)   Z 深度（用于排序）
    radii_2d: np.ndarray       # (N,)   2D 显示半径（已缩放）
    colors: list               # (N,)   颜色 hex
    symbols: list              # (N,)   元素符号
    numbers: np.ndarray        # (N,)   原子序数
    scale: float               # 缩放因子
    width: float               # 画布宽度
    height: float              # 画布高度
    rotation_matrix: np.ndarray  # (3, 3) 旋转矩阵
    cell_vertices_2d: object       # (8, 2) 或 None，晶胞顶点 2D 坐标
    cell_lines_2d: object          # 晶胞线段列表或 None
    source_atom_indices: np.ndarray  # (N,) 原始文件中的零基原子索引
    instance_keys: tuple[AtomInstanceKey, ...]  # (source, replica translation)
    image_shifts: np.ndarray  # (N, 3) 实际显示像整数平移
    instance_index_by_key: Mapping[AtomInstanceKey, int]
    cell_vectors_projected: object = None  # (3, 3) 旋转缩放后的晶胞向量
    outline_colors: list | None = None  # (N,) 每个原子的描边色
    color_strengths: np.ndarray | None = None  # (N,) 绝对色彩强度

    @property
    def natoms(self) -> int:
        return len(self.symbols)

    @property
    def z_order(self) -> np.ndarray:
        """按 Z 深度排序的原子索引（远→近）。"""
        return self.depths.argsort()


def project_atoms(atoms: Atoms, config: RenderConfig) -> ProjectionResult:
    """按默认单副本显示模型将 3D 原子构型投影到 2D。"""
    display = build_periodic_display(
        atoms,
        (),
        CellPeriodicSettings(show_unit_cell=config.show_unit_cell),
    )
    return project_periodic_display(atoms, display, config, frozenset())


def project_periodic_display(
    atoms: Atoms,
    display: PeriodicDisplay,
    config: RenderConfig,
    hidden_atom_indices: frozenset[int],
) -> ProjectionResult:
    """投影统一周期显示模型，并按原始原子身份应用样式与隐藏过滤。

    Parameters
    ----------
    atoms : Atoms
        ASE Atoms 对象
    display : PeriodicDisplay
        已由 matched 共价键构建的统一周期显示模型
    config : RenderConfig
        渲染参数
    hidden_atom_indices : frozenset[int]
        在显示模型构建后按原始索引过滤的隐藏原子

    Returns
    -------
    ProjectionResult
        包含 2D 坐标、Z 深度、半径、颜色等
    """
    if not isinstance(display, PeriodicDisplay):
        raise TypeError("周期投影必须使用 PeriodicDisplay")
    hidden = frozenset(hidden_atom_indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(atoms)
        for index in hidden
    ):
        raise ValueError("隐藏原子索引超出当前构型范围")

    # ── 1. 旋转矩阵 ──────────────────────────────────────────
    if config.rotation_matrix is not None:
        rotation_matrix = config.rotation_matrix
    else:
        rotation_matrix = rotate(config.rotation)

    # ── 2. 原子数据提取 ──────────────────────────────────────
    source_numbers = atoms.get_atomic_numbers()
    source_symbols = atoms.get_chemical_symbols()
    visible_instances = tuple(
        instance
        for instance in display.atom_instances
        if instance.source_atom_index not in hidden
    )
    source_atom_indices = np.asarray(
        [instance.source_atom_index for instance in visible_instances],
        dtype=int,
    )
    instance_keys = tuple(
        (instance.source_atom_index, instance.replica_translation)
        for instance in visible_instances
    )
    image_shifts = np.asarray(
        [instance.image_shift for instance in visible_instances],
        dtype=int,
    ).reshape((-1, 3))
    positions_3d = (
        np.vstack([instance.position for instance in visible_instances])
        if visible_instances
        else np.empty((0, 3), dtype=float)
    )
    natoms = len(visible_instances)
    numbers = source_numbers[source_atom_indices]
    symbols = [source_symbols[index] for index in source_atom_indices]

    # 最终显示半径由 RenderConfig 统一解析，周期副本继承源原子半径。
    source_radii = config.get_atom_radii(source_symbols)
    radii = source_radii[source_atom_indices]

    # 具体原子样式只计算一次，再按 source_atom_index 聚集到全部副本。
    source_colors = config.get_atom_colors(source_symbols)
    source_outline_colors = config.get_atom_outline_colors(len(atoms))
    source_color_strengths = config.get_atom_color_strengths(len(atoms))
    colors = [source_colors[index] for index in source_atom_indices]
    outline_colors = [source_outline_colors[index] for index in source_atom_indices]
    color_strengths = source_color_strengths[source_atom_indices]

    # ── 3. 晶胞处理 ──────────────────────────────────────────
    cell = atoms.get_cell()
    disp = atoms.get_celldisp().flatten()
    show_unit_cell = config.show_unit_cell

    if show_unit_cell > 0:
        cell_lines, cell_T, cell_D, cell_vertices_3d = _cell_to_lines(
            cell,
            disp,
            show_unit_cell,
            radii,
            positions_3d,
        )
    else:
        cell_lines = np.empty((0, 3))
        cell_T = None
        cell_D = None
        cell_vertices_3d = None

    # ── 4. 合并原子和晶胞线段，统一旋转 ──────────────────────
    nlines = len(cell_lines)
    all_positions = np.empty((natoms + nlines, 3))
    all_positions[:natoms] = positions_3d
    all_positions[natoms:] = cell_lines

    # 应用旋转
    all_positions = np.dot(all_positions, rotation_matrix)
    R = all_positions[:natoms]

    # ── 5. 计算 bbox、缩放、偏移 ────────────────────────────
    cell_vertices_rotated = (
        np.dot(cell_vertices_3d, rotation_matrix)
        if cell_vertices_3d is not None
        else None
    )
    bounds_min = []
    bounds_max = []
    if natoms:
        bounds_min.append((R - radii[:, None]).min(0))
        bounds_max.append((R + radii[:, None]).max(0))
    if cell_vertices_rotated is not None and (show_unit_cell == 2 or not natoms):
        bounds_min.append(cell_vertices_rotated.min(0))
        bounds_max.append(cell_vertices_rotated.max(0))

    if bounds_min:
        X1 = np.min(np.vstack(bounds_min), axis=0)
        X2 = np.max(np.vstack(bounds_max), axis=0)
    else:
        X1 = np.array([-0.5, -0.5, -0.5], dtype=float)
        X2 = np.array([0.5, 0.5, 0.5], dtype=float)

    M = (X1 + X2) / 2
    S = 1.05 * (X2 - X1)
    S[:2] = np.where(S[:2] > np.finfo(float).eps, S[:2], 1.0)
    scale = config.scale
    w = scale * S[0]
    if w > config.maxwidth:
        w = config.maxwidth
        scale = w / S[0]
    h = scale * S[1]

    offset = np.array([scale * M[0] - w / 2, scale * M[1] - h / 2, 0])

    # ── 6. 应用缩放和偏移 ────────────────────────────────────
    all_positions *= scale
    all_positions -= offset

    R_2d = all_positions[:natoms, :2].copy()
    depths = all_positions[:natoms, 2].copy()
    radii_2d = 2 * scale * radii / 2  # diameter → radius
    # 注意：ASE 中 self.d = 2 * scale * radii（直径）
    # 我们的 radii_2d 存储的是显示半径
    radii_2d = scale * radii
    cell_vectors_projected = np.dot(np.asarray(cell), rotation_matrix) * scale

    # 晶胞线段 2D
    cell_lines_2d = None
    if nlines > 0:
        cell_lines_2d_raw = all_positions[natoms:, :2].copy()
        D_2d = np.dot(cell_D, rotation_matrix)[:, :2] * scale if cell_D is not None else None
        cell_lines_2d = {
            'positions': cell_lines_2d_raw,
            'T': cell_T,
            'D': D_2d,
        }

    # 晶胞顶点 2D
    cell_vertices_2d = None
    if show_unit_cell == 2 and cell_vertices_rotated is not None:
        cell_vertices_2d = (cell_vertices_rotated * scale - offset)[:, :2]

    return ProjectionResult(
        positions_2d=R_2d,
        depths=depths,
        radii_2d=radii_2d,
        colors=colors,
        symbols=symbols,
        numbers=numbers,
        scale=scale,
        width=w,
        height=h,
        rotation_matrix=rotation_matrix,
        cell_vertices_2d=cell_vertices_2d,
        cell_lines_2d=cell_lines_2d,
        source_atom_indices=source_atom_indices,
        instance_keys=instance_keys,
        image_shifts=image_shifts,
        instance_index_by_key=MappingProxyType(
            {key: index for index, key in enumerate(instance_keys)}
        ),
        cell_vectors_projected=cell_vectors_projected,
        outline_colors=outline_colors,
        color_strengths=color_strengths,
    )


def _cell_to_lines(cell, disp, show_unit_cell, radii, R):
    """将晶胞边离散为线段（借鉴 ASE cell_to_lines）。

    返回 (positions, T, D, cell_vertices_3d)
    """
    from math import sqrt

    nlines = 0
    nsegments = []
    for c in range(3):
        d = sqrt((cell[c] ** 2).sum())
        n = max(2, int(d / 0.3))
        nsegments.append(n)
        nlines += 4 * n

    positions = np.empty((nlines, 3))
    T = np.empty(nlines, int)
    D = np.zeros((3, 3))

    n1 = 0
    for c in range(3):
        n = nsegments[c]
        dd = cell[c] / (4 * n - 2)
        D[c] = dd
        P = np.arange(1, 4 * n + 1, 4)[:, None] * dd
        T[n1:] = c
        for i, j in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            n2 = n1 + n
            positions[n1:n2] = P + i * cell[c - 2] + j * cell[c - 1]
            n1 = n2

    # 隐藏被原子完全覆盖的晶胞线段
    r2 = radii ** 2
    for n in range(nlines):
        d = D[T[n]]
        if ((((R - positions[n] - d) ** 2).sum(1) < r2) &
            (((R - positions[n] + d) ** 2).sum(1) < r2)).any():
            T[n] = -1

    # 晶胞顶点 3D
    cell_vertices = np.empty((2, 2, 2, 3))
    for c1 in range(2):
        for c2 in range(2):
            for c3 in range(2):
                cell_vertices[c1, c2, c3] = np.dot([c1, c2, c3], cell) + disp
    cell_vertices = cell_vertices.reshape(8, 3)

    return positions, T, D, cell_vertices
