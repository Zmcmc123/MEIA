"""
配置参数模块。
"""

import math

import numpy as np
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Dict, Mapping, Optional, Set, Tuple


RECOMMENDED_COLORS: Dict[str, str] = {
    "H":  "#E6E6E5",
    "C":  "#3F4F6A",
    "O":  "#E5A6A6",
    "Si": "#5386C6",
    "Ca": "#9ECC91",
}

DEFAULT_STROKE_COLOR = "#231815"

DEFAULT_ALLOWED_PAIRS: Set[Tuple[str, str]] = {
    ("H", "O"),
    ("C", "O"),
    ("C", "H"),
    ("C", "C"),
    ("O", "Si"),
    ("Ca", "O"),
    ("O", "O"),
    ("Si", "Si"),
}


@dataclass
class RenderConfig:
    """渲染参数配置。"""

    # ── 原子渲染 ──────────────────────────────────────────────
    radius_scale: float = 0.6
    outline_width: float = 0.5
    """原子轮廓粗细"""

    # ── 化学键渲染 ────────────────────────────────────────────
    bond_cutoff: float = 1.0
    bond_width_ratio: float = 0.45
    """键宽 / 平均原子显示半径"""

    ellipse_ry_ratio: float = 0.30
    """穹顶深度 = bond_width * ratio（沿键方向的短轴）"""

    ellipse_atom_scale: bool = False
    """兼容旧配置；v0.5 起椭圆帽只随键宽缩放，不再按原子半径缩放。"""

    ellipse_min_scale: float = 0.5
    """兼容旧配置；固定比例椭圆帽不再使用此参数。"""

    # ── 描边 ──────────────────────────────────────────────────
    stroke_color: str = DEFAULT_STROKE_COLOR
    bond_stroke_color: Optional[str] = None
    """化学键专用描边颜色；为 None 时沿用原子轮廓颜色。"""
    bond_stroke_width: float = 0.25
    """化学键椭圆帽与平行线共用的描边线宽"""

    # ── 导出 ──────────────────────────────────────────────────
    transparent: bool = True
    dpi: int = 600

    # ── 视角 ──────────────────────────────────────────────────
    rotation: str = "-90x"

    rotation_matrix: Optional["np.ndarray"] = None
    """直接旋转矩阵；设置后覆盖 rotation 字符串（用于 Plotly 相机视角转换）"""
    show_unit_cell: int = 2

    # ── 颜色 ──────────────────────────────────────────────────
    custom_colors: Optional[Dict[str, str]] = field(default=None)

    scale: float = 1.0
    maxwidth: float = 500

    allowed_pairs: Optional[Set[Tuple[str, str]]] = None
    atom_color_strengths: Mapping[int, float] = field(default_factory=dict)
    atom_color_overrides: Mapping[int, str] = field(default_factory=dict)
    # 新增字段只追加到末尾，保持旧位置参数的含义不变。
    resolved_element_radii_angstrom: Optional[Mapping[str, float]] = None

    def __post_init__(self) -> None:
        if self.resolved_element_radii_angstrom is not None:
            from ase.data import atomic_numbers

            if not isinstance(self.resolved_element_radii_angstrom, Mapping):
                raise TypeError("已解析元素半径必须是映射")
            normalized: dict[str, float] = {}
            for symbol, radius in self.resolved_element_radii_angstrom.items():
                if symbol not in atomic_numbers or symbol == "X":
                    raise ValueError(f"resolved element radii invalid symbol: {symbol!r}")
                if (
                    isinstance(radius, bool)
                    or not isinstance(radius, (int, float, np.number))
                    or not math.isfinite(radius)
                    or radius <= 0
                ):
                    raise ValueError(f"resolved element radius for {symbol} must be > 0 and finite")
                normalized[symbol] = float(radius)
            object.__setattr__(
                self,
                "resolved_element_radii_angstrom",
                MappingProxyType(normalized),
            )

    # 兼容旧代码的属性
    @property
    def stroke_width(self):
        return self.bond_stroke_width

    @property
    def effective_bond_stroke_color(self):
        return self.bond_stroke_color or self.stroke_color

    @property
    def ellipse_rx_ratio(self):
        return 0.5  # 长轴始终 = bond_width，此值仅用于兼容

    def get_effective_allowed_pairs(self):
        if self.allowed_pairs is not None:
            return self.allowed_pairs if self.allowed_pairs else None
        return DEFAULT_ALLOWED_PAIRS

    def get_atom_colors(self, symbols) -> list:
        from matplotlib.colors import to_hex
        from ase.data import atomic_numbers
        from ase.data.colors import jmol_colors
        from .atom_styles import apply_color_strength

        colors = self.custom_colors or RECOMMENDED_COLORS
        result = []
        strengths = self.get_atom_color_strengths(len(symbols))
        specific_colors = self.get_atom_color_overrides(len(symbols))
        for index, sym in enumerate(symbols):
            if index in specific_colors:
                base = specific_colors[index]
            elif sym in colors:
                base = colors[sym]
            else:
                Z = atomic_numbers[sym]
                base = to_hex(jmol_colors[Z])
            result.append(apply_color_strength(base, strengths[index]))
        return result

    def get_atom_radii(self, symbols) -> np.ndarray:
        """返回最终原子显示半径；显式解析映射优先于兼容半径倍率。"""
        from ase.data import atomic_numbers, covalent_radii

        symbol_tuple = tuple(symbols)
        if not symbol_tuple:
            return np.empty(0, dtype=float)
        if self.resolved_element_radii_angstrom is None:
            numbers = np.asarray(
                [atomic_numbers[symbol] for symbol in symbol_tuple], dtype=int
            )
            return covalent_radii[numbers] * self.radius_scale
        try:
            return np.asarray(
                [float(self.resolved_element_radii_angstrom[symbol]) for symbol in symbol_tuple],
                dtype=float,
            )
        except KeyError as exc:
            raise ValueError(
                f"resolved element radii missing symbol: {exc.args[0]!r}"
            ) from exc

    def get_atom_color_overrides(self, atom_count: int) -> dict[int, str]:
        from matplotlib.colors import is_color_like, to_hex

        values: dict[int, str] = {}
        for atom_index, color in self.atom_color_overrides.items():
            if (
                isinstance(atom_index, bool)
                or not isinstance(atom_index, int)
                or not 0 <= atom_index < atom_count
            ):
                raise ValueError(f"具体原子颜色索引越界：{atom_index!r}")
            if not isinstance(color, str) or not is_color_like(color):
                raise ValueError(f"非法具体原子颜色：{color!r}")
            values[atom_index] = to_hex(color).upper()
        return values

    def get_atom_color_strengths(self, atom_count: int) -> np.ndarray:
        from .atom_styles import normalize_color_strength

        values = np.ones(atom_count, dtype=float)
        for atom_index, strength in self.atom_color_strengths.items():
            if (
                isinstance(atom_index, bool)
                or not isinstance(atom_index, int)
                or not 0 <= atom_index < atom_count
            ):
                raise ValueError(f"原子色彩强度索引越界：{atom_index!r}")
            values[atom_index] = normalize_color_strength(strength)
        return values

    def get_atom_outline_colors(self, atom_count: int) -> list[str]:
        from .atom_styles import apply_color_strength

        return [
            apply_color_strength(self.stroke_color, strength)
            for strength in self.get_atom_color_strengths(atom_count)
        ]
