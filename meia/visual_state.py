"""可视化已应用状态与 2D/3D/导出共用的统一渲染上下文。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import combinations_with_replacement
import math
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.utils import rotate
from matplotlib.colors import is_color_like, to_hex

from .atom_styles import (
    AtomSelectionSettings,
    validate_atom_selection_settings,
)
from .bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondResolution,
    BondSettings,
    BondStrokeStyle,
    BondStyle,
    OverrideVisibility,
    default_pair_max_distance,
    normalize_element_pair,
    reapply_bond_visibility,
    resolve_bonds,
)
from .config import RECOMMENDED_COLORS, RenderConfig
from .hydrogen_bonds import (
    DisplayHydrogenBond,
    HydrogenBondSettings,
)
from .i18n import LocalizedError
from .periodic_display import (
    CellPeriodicSettings,
    PeriodicDisplay,
    normalize_periodic_settings,
)
from .size_profiles import (
    SizeProfileSettings,
    resolve_active_bond_width,
)
from .view_state import (
    CameraState,
    rotation_matrix_to_camera,
)


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{label}必须是数值")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label}必须是有限数值")
    return result


def _default_camera() -> CameraState:
    return rotation_matrix_to_camera(np.asarray(rotate("-90x"), dtype=float))


@dataclass(frozen=True)
class PairRuleDefaults:
    """自动生成缺失元素对时使用的距离默认策略。"""

    bond_cutoff: float = 1.0
    long_distance_threshold_angstrom: float = 2.0
    pair_distance_multipliers: tuple[tuple[str, str, float], ...] = (
        ("H", "O", 1.20),
    )

    def __post_init__(self) -> None:
        cutoff = _finite_number(self.bond_cutoff, "成键阈值")
        if cutoff <= 0:
            raise ValueError("成键阈值必须大于 0")
        threshold = _finite_number(
            self.long_distance_threshold_angstrom,
            "长距离键默认阈值",
        )
        if threshold <= 0:
            raise ValueError("长距离键默认阈值必须大于 0 Å")
        canonical: dict[tuple[str, str], float] = {}
        for item in self.pair_distance_multipliers:
            if not isinstance(item, (tuple, list)) or len(item) != 3:
                raise ValueError("元素对距离乘数必须是（元素，元素，乘数）")
            pair = normalize_element_pair(item[0], item[1])
            multiplier = _finite_number(item[2], "元素对距离乘数")
            if multiplier <= 0:
                raise ValueError("元素对距离乘数必须大于 0")
            if pair in canonical:
                raise ValueError(f"元素对距离乘数重复：{pair}")
            canonical[pair] = multiplier
        object.__setattr__(self, "bond_cutoff", cutoff)
        object.__setattr__(self, "long_distance_threshold_angstrom", threshold)
        object.__setattr__(
            self,
            "pair_distance_multipliers",
            tuple((a, b, value) for (a, b), value in sorted(canonical.items())),
        )

    def multiplier_mapping(self) -> Mapping[tuple[str, str], float]:
        return MappingProxyType(
            {(a, b): value for a, b, value in self.pair_distance_multipliers}
        )

@dataclass(frozen=True)
class ViewSettings:
    """视角表单已应用状态。"""

    rotation: str = "-90x"
    camera: CameraState = field(default_factory=_default_camera)

    def __post_init__(self) -> None:
        if not isinstance(self.rotation, str) or not self.rotation.strip():
            raise ValueError("视角旋转必须是非空字符串")
        try:
            matrix = np.asarray(rotate(self.rotation), dtype=float)
        except Exception as exc:
            raise ValueError(f"非法视角旋转：{self.rotation!r}") from exc
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"非法视角旋转：{self.rotation!r}")
        if not isinstance(self.camera, CameraState):
            raise TypeError("已应用视角必须包含 CameraState")


@dataclass(frozen=True)
class AtomCellSettings:
    """原子全局样式与元素配色。"""

    outline_width: float = 0.5
    element_colors: Mapping[str, str] = field(
        default_factory=lambda: dict(RECOMMENDED_COLORS)
    )

    def __post_init__(self) -> None:
        outline = _finite_number(self.outline_width, "原子描边粗细")
        if outline < 0:
            raise LocalizedError(
                "原子描边粗细不能小于 0",
                message_key="atom.outline_nonnegative",
                message_params={"value": outline},
            )
        if not isinstance(self.element_colors, Mapping):
            raise TypeError("元素配色必须是映射")
        colors: dict[str, str] = {}
        for symbol, color in self.element_colors.items():
            if symbol not in atomic_numbers:
                raise LocalizedError(
                    f"元素配色包含非法元素：{symbol!r}",
                    message_key="atom.invalid_element",
                    message_params={"symbol": repr(symbol)},
                )
            if not isinstance(color, str) or not is_color_like(color):
                raise LocalizedError(
                    f"元素 {symbol} 的颜色无效：{color!r}",
                    message_key="atom.invalid_element_color",
                    message_params={"symbol": symbol, "value": repr(color)},
                )
            colors[symbol] = to_hex(color).upper()
        object.__setattr__(self, "outline_width", outline)
        object.__setattr__(self, "element_colors", MappingProxyType(colors))


@dataclass(frozen=True)
class BondModuleSettings:
    """化学键表单的全局已应用状态。"""

    draw_bonds: bool = True
    style: BondStrokeStyle = field(default_factory=BondStrokeStyle)
    defaults: PairRuleDefaults = field(default_factory=PairRuleDefaults)
    pair_rules: tuple[BondPairRule, ...] = ()
    hydrogen_bonds: HydrogenBondSettings = field(
        default_factory=HydrogenBondSettings
    )

    def __post_init__(self) -> None:
        if not isinstance(self.draw_bonds, bool):
            raise ValueError("化学键总开关必须是布尔值")
        if not isinstance(self.style, BondStrokeStyle):
            raise TypeError("化学键描边样式必须是 BondStrokeStyle")
        if not isinstance(self.defaults, PairRuleDefaults):
            raise TypeError("化学键默认策略必须是 PairRuleDefaults")
        if not isinstance(self.hydrogen_bonds, HydrogenBondSettings):
            raise TypeError("氢键设置必须是 HydrogenBondSettings")
        rules = tuple(self.pair_rules)
        if not all(isinstance(rule, BondPairRule) for rule in rules):
            raise TypeError("元素对规则必须由 BondPairRule 组成")
        pairs = [rule.pair for rule in rules]
        if len(pairs) != len(set(pairs)):
            raise ValueError("元素对规则重复")
        object.__setattr__(self, "pair_rules", rules)


@dataclass(frozen=True)
class ExportSettings:
    """文件导出与画布背景已应用状态。"""

    format: str = "svg"
    dpi: int = 600
    transparent: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.format, str) or self.format.lower() not in {
            "svg",
            "png",
            "pdf",
        }:
            raise ValueError(f"不支持的导出格式：{self.format!r}")
        if (
            isinstance(self.dpi, bool)
            or not isinstance(self.dpi, int)
            or self.dpi <= 0
        ):
            raise ValueError("DPI 必须是大于 0 的整数")
        if not isinstance(self.transparent, bool):
            raise ValueError("透明背景必须是布尔值")
        object.__setattr__(self, "format", self.format.lower())


@dataclass(frozen=True)
class PortableStyle:
    """不绑定具体结构的通用可视化风格。"""

    view: ViewSettings = field(default_factory=ViewSettings)
    size_profiles: SizeProfileSettings = field(default_factory=SizeProfileSettings)
    atom_cell: AtomCellSettings = field(default_factory=AtomCellSettings)
    bonds: BondModuleSettings = field(default_factory=BondModuleSettings)
    cell_periodic: CellPeriodicSettings = field(default_factory=CellPeriodicSettings)
    export: ExportSettings = field(default_factory=ExportSettings)


@dataclass(frozen=True)
class VisualizationState:
    """可移植风格与具体原子状态的完整已应用状态。"""

    style: PortableStyle = field(default_factory=PortableStyle)
    atom_selection: AtomSelectionSettings = field(
        default_factory=AtomSelectionSettings
    )


@dataclass(frozen=True)
class RenderContext:
    """2D、3D 与导出入口共用的唯一渲染参数。"""

    config: RenderConfig
    bond_settings: BondSettings
    bond_resolution: BondResolution
    periodic_topology_bonds: tuple[ResolvedBond, ...]
    periodic_display: PeriodicDisplay
    hydrogen_bonds: tuple[DisplayHydrogenBond, ...]
    hidden_atom_indices: frozenset[int]


def _generated_pair_rule(
    pair: tuple[str, str], defaults: PairRuleDefaults
) -> BondPairRule:
    return BondPairRule(
        pair[0],
        pair[1],
        min_distance=0.0,
        max_distance=default_pair_max_distance(
            pair[0],
            pair[1],
            bond_cutoff=defaults.bond_cutoff,
            pair_distance_multipliers=defaults.multiplier_mapping(),
        ),
        enabled=True,
        participates_in_periodic_unwrap=True,
    )


def _classify_generated_pair_rules(
    candidate_rules: Iterable[BondPairRule],
    resolution: BondResolution,
    defaults: PairRuleDefaults,
) -> tuple[BondPairRule, ...]:
    minimum_by_pair: dict[tuple[str, str], float] = {}
    for bond in resolution.matched:
        minimum_by_pair[bond.pair] = min(
            bond.distance,
            minimum_by_pair.get(bond.pair, bond.distance),
        )
    return tuple(
        replace(
            rule,
            enabled=minimum_by_pair[rule.pair]
            <= defaults.long_distance_threshold_angstrom,
            participates_in_periodic_unwrap=minimum_by_pair[rule.pair]
            <= defaults.long_distance_threshold_angstrom,
        )
        for rule in candidate_rules
        if rule.pair in minimum_by_pair
    )


def merge_pair_rules_for_structure(
    atoms: Atoms,
    bonds: BondModuleSettings,
) -> BondModuleSettings:
    """保留显式规则，并为当前结构中实际匹配的缺失元素对补齐默认规则。"""
    if not isinstance(bonds, BondModuleSettings):
        raise TypeError("化学键模块设置必须是 BondModuleSettings")
    explicit_pairs = {rule.pair for rule in bonds.pair_rules}
    present = sorted(set(atoms.get_chemical_symbols()))
    generated = tuple(
        _generated_pair_rule(pair, bonds.defaults)
        for pair in combinations_with_replacement(present, 2)
        if pair not in explicit_pairs
    )
    resolution = resolve_bonds(atoms, BondSettings(pair_rules=generated))
    additions = _classify_generated_pair_rules(
        generated,
        resolution,
        bonds.defaults,
    )
    return replace(bonds, pair_rules=bonds.pair_rules + additions)


def merge_portable_style_for_structure(
    style: PortableStyle,
    atoms: Atoms,
) -> PortableStyle:
    if not isinstance(style, PortableStyle):
        raise TypeError("通用风格必须是 PortableStyle")
    return replace(style, bonds=merge_pair_rules_for_structure(atoms, style.bonds))


def _ensure_override_pairs(
    bonds: BondModuleSettings,
    overrides: Iterable[AtomBondOverride],
) -> BondModuleSettings:
    existing = {rule.pair for rule in bonds.pair_rules}
    missing = sorted({item.pair for item in overrides} - existing)
    if not missing:
        return bonds
    additions = tuple(_generated_pair_rule(pair, bonds.defaults) for pair in missing)
    return replace(bonds, pair_rules=bonds.pair_rules + additions)


def _resolve_context_bonds(
    atoms: Atoms,
    bonds: BondModuleSettings,
    overrides: Iterable[AtomBondOverride],
    bond_width_ratio: float,
) -> tuple[BondModuleSettings, BondSettings, BondResolution]:
    """一次解析候选规则，并仅保留显式、匹配或具体例外元素对。"""
    overrides = tuple(overrides)
    explicit_pairs = {rule.pair for rule in bonds.pair_rules}
    present = sorted(set(atoms.get_chemical_symbols()))
    generated = tuple(
        _generated_pair_rule(pair, bonds.defaults)
        for pair in combinations_with_replacement(present, 2)
        if pair not in explicit_pairs
    )
    candidates = _ensure_override_pairs(
        replace(bonds, pair_rules=bonds.pair_rules + generated),
        overrides,
    )
    render_style = BondStyle(
        width_ratio=bond_width_ratio,
        stroke_width=candidates.style.stroke_width,
        stroke_color=candidates.style.stroke_color,
    )
    candidate_settings = BondSettings(
        draw_bonds=candidates.draw_bonds,
        pair_rules=candidates.pair_rules,
        atom_overrides=overrides,
        style=render_style,
    )
    resolution = resolve_bonds(atoms, candidate_settings)
    classified_generated = _classify_generated_pair_rules(
        generated,
        resolution,
        bonds.defaults,
    )
    classified_pairs = {rule.pair for rule in classified_generated}
    override_pairs = {item.pair for item in overrides}
    override_only_pairs = override_pairs - explicit_pairs - classified_pairs
    resolved_bonds = replace(
        candidates,
        pair_rules=(
            bonds.pair_rules
            + classified_generated
            + tuple(
                rule
                for rule in candidates.pair_rules
                if rule.pair in override_only_pairs
            )
        ),
    )
    settings = BondSettings(
        draw_bonds=resolved_bonds.draw_bonds,
        pair_rules=resolved_bonds.pair_rules,
        atom_overrides=overrides,
        style=BondStyle(
            width_ratio=bond_width_ratio,
            stroke_width=resolved_bonds.style.stroke_width,
            stroke_color=resolved_bonds.style.stroke_color,
        ),
    )
    return resolved_bonds, settings, reapply_bond_visibility(resolution, settings)


def resolve_render_context(
    atoms: Atoms,
    state: VisualizationState,
) -> RenderContext:
    """一次性解析所有已应用模块，供所有渲染入口共用。"""
    from .render_topology import build_render_topology, compose_render_context

    topology = build_render_topology(atoms, state)
    return compose_render_context(atoms, state, topology)


def replace_view(state: VisualizationState, view: ViewSettings) -> VisualizationState:
    return replace(state, style=replace(state.style, view=view))


def replace_atom_cell(
    state: VisualizationState, atom_cell: AtomCellSettings
) -> VisualizationState:
    return replace(state, style=replace(state.style, atom_cell=atom_cell))


def replace_atom_and_size_profiles(
    state: VisualizationState,
    atom_cell: AtomCellSettings,
    size_profiles: SizeProfileSettings,
) -> VisualizationState:
    """原子化替换原子样式与其耦合的尺寸档案。"""
    return replace(
        state,
        style=replace(
            state.style,
            atom_cell=atom_cell,
            size_profiles=size_profiles,
        ),
    )


def replace_bonds(
    state: VisualizationState, bonds: BondModuleSettings
) -> VisualizationState:
    return replace(state, style=replace(state.style, bonds=bonds))


def replace_bonds_and_size_profiles(
    state: VisualizationState,
    bonds: BondModuleSettings,
    size_profiles: SizeProfileSettings,
) -> VisualizationState:
    """原子化替换化学键设置与当前档案的键宽。"""
    return replace(
        state,
        style=replace(
            state.style,
            bonds=bonds,
            size_profiles=size_profiles,
        ),
    )


def replace_cell_periodic(
    state: VisualizationState, cell_periodic: CellPeriodicSettings
) -> VisualizationState:
    return replace(state, style=replace(state.style, cell_periodic=cell_periodic))


def replace_atom_selection(
    state: VisualizationState, atom_selection: AtomSelectionSettings
) -> VisualizationState:
    return replace(state, atom_selection=atom_selection)


def replace_export(
    state: VisualizationState, export: ExportSettings
) -> VisualizationState:
    return replace(state, style=replace(state.style, export=export))


def reset_visual_modules_from_style(
    current: VisualizationState,
    baseline: PortableStyle,
    atoms: Atoms,
) -> VisualizationState:
    """以通用风格基准原子化还原视觉模块，保留视角与导出设置。"""
    if not isinstance(current, VisualizationState):
        raise TypeError("当前可视化状态必须是 VisualizationState")
    if not isinstance(baseline, PortableStyle):
        raise TypeError("还原基准必须是 PortableStyle")
    normalized_baseline = replace(
        baseline,
        cell_periodic=normalize_periodic_settings(atoms, baseline.cell_periodic),
    )
    candidate = merge_portable_style_for_structure(normalized_baseline, atoms)
    candidate = replace(
        candidate,
        view=current.style.view,
        export=current.style.export,
    )
    reset_state = VisualizationState(
        style=candidate,
        atom_selection=AtomSelectionSettings(),
    )
    resolve_render_context(atoms, reset_state)
    return reset_state


def apply_camera_only(
    state: VisualizationState,
    camera: CameraState,
) -> VisualizationState:
    """只替换已应用相机，不读取或提交任何其他模块。"""
    if not isinstance(camera, CameraState):
        raise TypeError("当前视角必须是 CameraState")
    return replace_view(state, replace(state.style.view, camera=camera))


def apply_portable_style(
    state: VisualizationState,
    style: PortableStyle,
    atoms: Atoms,
) -> VisualizationState:
    """应用通用风格，保留当前所有具体原子状态。"""
    merged = merge_portable_style_for_structure(style, atoms)
    merged = replace(
        merged,
        cell_periodic=normalize_periodic_settings(atoms, merged.cell_periodic),
    )
    merged = replace(
        merged,
        bonds=_ensure_override_pairs(merged.bonds, state.atom_selection.bond_overrides),
    )
    validate_atom_selection_settings(
        atoms,
        state.atom_selection,
        tuple(rule.pair for rule in merged.bonds.pair_rules),
    )
    return replace(state, style=merged)
