"""VESTA 风格的元素对成键规则与具体原子可见性例外。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from itertools import combinations_with_replacement
import math
from types import MappingProxyType
from typing import Mapping, Sequence, Tuple

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, covalent_radii
from ase.neighborlist import neighbor_list
from matplotlib.colors import is_color_like
from .i18n import LocalizedError

from .bond_segments import clip_bond_to_spheres
from .config import DEFAULT_STROKE_COLOR, RenderConfig


ElementPair = Tuple[str, str]
DEFAULT_PAIR_DISTANCE_MULTIPLIERS: Mapping[ElementPair, float] = MappingProxyType(
    {("H", "O"): 1.20}
)


class BondRuleError(LocalizedError):
    """化学键规则无效。"""

    def __init__(
        self,
        technical_message: str,
        *,
        message_key: str = "bonds.invalid_setting",
        message_params: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            technical_message,
            message_key=message_key,
            message_params=message_params,
        )


class OverrideVisibility(str, Enum):
    """具体原子对某类键的可见性覆盖。"""

    INHERIT = "inherit"
    SHOW = "show"
    HIDE = "hide"


def _validate_element(symbol: str) -> str:
    if not isinstance(symbol, str) or symbol not in atomic_numbers:
        raise BondRuleError(
            f"非法元素符号：{symbol!r}",
            message_key="bonds.invalid_element",
            message_params={"symbol": repr(symbol)},
        )
    return symbol


def normalize_element_pair(element_a: str, element_b: str) -> ElementPair:
    """把元素对规范化为与输入端点顺序无关的稳定元组。"""
    a = _validate_element(element_a)
    b = _validate_element(element_b)
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def default_pair_max_distance(
    element_a: str,
    element_b: str,
    *,
    bond_cutoff: float = 1.0,
    pair_distance_multipliers: Mapping[ElementPair, float] | None = None,
) -> float:
    """返回元素对的默认最大距离，单位为 Å。"""
    pair = normalize_element_pair(element_a, element_b)
    multipliers = pair_distance_multipliers or DEFAULT_PAIR_DISTANCE_MULTIPLIERS
    multiplier = multipliers.get(pair, 1.0)
    return float(
        (
            covalent_radii[atomic_numbers[pair[0]]]
            + covalent_radii[atomic_numbers[pair[1]]]
        )
        * bond_cutoff
        * multiplier
    )


def is_primary_cell_offset(offset: Sequence[int]) -> bool:
    """仅当键的两个端点原子都属于当前显示单胞时返回 True。"""
    return all(int(value) == 0 for value in offset)


def _finite_number(value: object, label: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise BondRuleError(
            f"{label}必须是数值",
            message_key="bonds.value_numeric",
            message_params={"field": field, "value": repr(value)},
        )
    result = float(value)
    if not math.isfinite(result):
        raise BondRuleError(
            f"{label}必须是有限数值",
            message_key="bonds.value_finite",
            message_params={"field": field, "value": repr(value)},
        )
    return result


@dataclass(frozen=True)
class BondPairRule:
    """一种元素对的闭区间距离规则，单位为 Å。"""

    element_a: str
    element_b: str
    min_distance: float
    max_distance: float
    enabled: bool = True
    participates_in_periodic_unwrap: bool = True

    def __post_init__(self) -> None:
        pair = normalize_element_pair(self.element_a, self.element_b)
        minimum = _finite_number(self.min_distance, "最小距离", "minimum distance")
        maximum = _finite_number(self.max_distance, "最大距离", "maximum distance")
        if minimum < 0:
            raise BondRuleError(
                "最小距离不能小于 0 Å",
                message_key="bonds.minimum_nonnegative",
                message_params={"value": minimum},
            )
        if maximum < minimum:
            raise BondRuleError(
                f"最大距离不能小于最小距离；收到 minimum={minimum!r}, "
                f"maximum={maximum!r}",
                message_key="bonds.range_invalid",
                message_params={"minimum": minimum, "maximum": maximum},
            )
        if not isinstance(self.enabled, bool):
            raise BondRuleError("元素对显示开关必须是布尔值")
        if not isinstance(self.participates_in_periodic_unwrap, bool):
            raise BondRuleError("元素对周期整理开关必须是布尔值")
        object.__setattr__(self, "element_a", pair[0])
        object.__setattr__(self, "element_b", pair[1])
        object.__setattr__(self, "min_distance", minimum)
        object.__setattr__(self, "max_distance", maximum)

    @property
    def pair(self) -> ElementPair:
        return normalize_element_pair(self.element_a, self.element_b)


@dataclass(frozen=True)
class AtomBondOverride:
    """某个具体原子对一种元素对规则的可见性覆盖。"""

    atom_index: int
    atom_symbol: str
    element_a: str
    element_b: str
    visibility: OverrideVisibility = OverrideVisibility.INHERIT

    def __post_init__(self) -> None:
        if isinstance(self.atom_index, bool) or not isinstance(self.atom_index, int):
            raise BondRuleError("具体原子索引必须是整数")
        if self.atom_index < 0:
            raise BondRuleError("具体原子索引不能小于 0")
        _validate_element(self.atom_symbol)
        pair = normalize_element_pair(self.element_a, self.element_b)
        try:
            visibility = OverrideVisibility(self.visibility)
        except ValueError as exc:
            raise BondRuleError(f"非法原子例外状态：{self.visibility!r}") from exc
        object.__setattr__(self, "element_a", pair[0])
        object.__setattr__(self, "element_b", pair[1])
        object.__setattr__(self, "visibility", visibility)

    @property
    def pair(self) -> ElementPair:
        return normalize_element_pair(self.element_a, self.element_b)


@dataclass(frozen=True)
class BondStyle:
    """所有化学键共用的显示样式。"""

    width_ratio: float = 0.45
    stroke_width: float = 0.25
    stroke_color: str = DEFAULT_STROKE_COLOR

    def __post_init__(self) -> None:
        width = _finite_number(self.width_ratio, "键宽比例", "bond width ratio")
        stroke = _finite_number(self.stroke_width, "键描边粗细", "bond outline width")
        if width <= 0:
            raise BondRuleError(
                "键宽比例必须大于 0",
                message_key="bonds.width_positive",
                message_params={"value": width},
            )
        if stroke < 0:
            raise BondRuleError(
                "键描边粗细不能小于 0",
                message_key="bonds.stroke_nonnegative",
                message_params={"value": stroke},
            )
        if not isinstance(self.stroke_color, str) or not is_color_like(self.stroke_color):
            raise BondRuleError(
                f"非法键描边颜色：{self.stroke_color!r}",
                message_key="bonds.invalid_stroke_color",
                message_params={"value": repr(self.stroke_color)},
            )
        object.__setattr__(self, "width_ratio", width)
        object.__setattr__(self, "stroke_width", stroke)


@dataclass(frozen=True)
class BondStrokeStyle:
    """可持久化的化学键描边样式；键体宽度由尺寸档案独立保存。"""

    stroke_width: float = 0.25
    stroke_color: str = DEFAULT_STROKE_COLOR

    def __post_init__(self) -> None:
        stroke = _finite_number(self.stroke_width, "键描边粗细", "bond outline width")
        if stroke < 0:
            raise BondRuleError(
                "键描边粗细不能小于 0",
                message_key="bonds.stroke_nonnegative",
                message_params={"value": stroke},
            )
        if not isinstance(self.stroke_color, str) or not is_color_like(self.stroke_color):
            raise BondRuleError(
                f"非法键描边颜色：{self.stroke_color!r}",
                message_key="bonds.invalid_stroke_color",
                message_params={"value": repr(self.stroke_color)},
            )
        object.__setattr__(self, "stroke_width", stroke)


@dataclass(frozen=True)
class BondSettings:
    """已应用的化学键规则与统一样式。"""

    draw_bonds: bool = True
    pair_rules: Sequence[BondPairRule] = ()
    atom_overrides: Sequence[AtomBondOverride] = ()
    style: BondStyle = field(default_factory=BondStyle)

    def __post_init__(self) -> None:
        if not isinstance(self.draw_bonds, bool):
            raise BondRuleError("化学键总开关必须是布尔值")
        rules = tuple(self.pair_rules)
        overrides = tuple(self.atom_overrides)
        rule_pairs = [rule.pair for rule in rules]
        if len(rule_pairs) != len(set(rule_pairs)):
            raise BondRuleError("元素对规则重复")
        override_keys = [(item.atom_index, item.pair) for item in overrides]
        if len(override_keys) != len(set(override_keys)):
            raise BondRuleError("具体原子例外规则重复")
        missing_pairs = sorted({item.pair for item in overrides} - set(rule_pairs))
        if missing_pairs:
            raise BondRuleError(f"具体原子例外缺少对应元素对规则：{missing_pairs}")
        object.__setattr__(self, "pair_rules", rules)
        object.__setattr__(self, "atom_overrides", overrides)


@dataclass(frozen=True)
class ResolvedBond:
    """元素对距离匹配完成后的稳定键记录。"""

    i: int
    j: int
    offset: Tuple[int, int, int]
    distance: float
    pair: ElementPair
    bond_id: str
    visible: bool
    visibility_source: str


@dataclass(frozen=True)
class BondResolution:
    """距离匹配结果和最终可见子集。"""

    matched: Tuple[ResolvedBond, ...]
    visible: Tuple[ResolvedBond, ...]
    match_counts: Mapping[ElementPair, int]


@dataclass(frozen=True)
class BondOverrideConflict:
    """同一根键两端分别强制隐藏与强制显示的可解释记录。"""

    bond_id: str
    atom_i: int
    atom_j: int
    element_pair: ElementPair
    hidden_atom_index: int
    shown_atom_index: int


def validate_bond_settings(atoms: Atoms, settings: BondSettings) -> None:
    """校验具体原子例外与当前构型一致。"""
    symbols = atoms.get_chemical_symbols()
    for override in settings.atom_overrides:
        if override.atom_index >= len(atoms):
            raise BondRuleError(
                f"具体原子索引越界：{override.atom_index}，当前共 {len(atoms)} 个原子"
            )
        actual = symbols[override.atom_index]
        if actual != override.atom_symbol:
            raise BondRuleError(
                f"具体原子元素不一致：索引 {override.atom_index} "
                f"应为 {actual}，规则中为 {override.atom_symbol}"
            )
        if actual not in override.pair:
            raise BondRuleError(
                f"具体原子 {actual} 不属于元素对 {override.pair[0]}–{override.pair[1]}"
            )


def initialize_bond_settings(
    atoms: Atoms,
    config: RenderConfig,
    *,
    long_distance_threshold_angstrom: float = 2.0,
) -> BondSettings:
    """按元素对默认阈值，仅为当前实际匹配的类型建立规则。"""
    threshold = _finite_number(
        long_distance_threshold_angstrom,
        "长距离键默认阈值",
        "long-distance bond threshold",
    )
    if threshold <= 0:
        raise BondRuleError("长距离键默认阈值必须大于 0 Å")
    allowed = config.get_effective_allowed_pairs()
    symbols = atoms.get_chemical_symbols()
    present = set(symbols)
    candidate_pairs = (
        {
            normalize_element_pair(element_a, element_b)
            for element_a, element_b in allowed
            if element_a in present and element_b in present
        }
        if allowed is not None
        else set(combinations_with_replacement(sorted(present), 2))
    )
    candidate_rules = tuple(
        BondPairRule(
            element_a,
            element_b,
            min_distance=0.0,
            max_distance=default_pair_max_distance(
                element_a,
                element_b,
                bond_cutoff=config.bond_cutoff,
            ),
            enabled=True,
            participates_in_periodic_unwrap=True,
        )
        for element_a, element_b in sorted(candidate_pairs)
    )
    candidate_settings = BondSettings(pair_rules=candidate_rules)
    resolution = resolve_bonds(atoms, candidate_settings)
    minimum_by_pair: dict[ElementPair, float] = {}
    for bond in resolution.matched:
        minimum_by_pair[bond.pair] = min(
            bond.distance,
            minimum_by_pair.get(bond.pair, bond.distance),
        )
    rules = tuple(
        replace(
            rule,
            enabled=minimum_by_pair[rule.pair] <= threshold,
            participates_in_periodic_unwrap=minimum_by_pair[rule.pair] <= threshold,
        )
        for rule in candidate_rules
        if rule.pair in minimum_by_pair
    )
    return BondSettings(
        draw_bonds=True,
        pair_rules=rules,
        style=BondStyle(
            width_ratio=config.bond_width_ratio,
            stroke_width=config.bond_stroke_width,
            stroke_color=config.effective_bond_stroke_color,
        ),
    )


def _canonical_candidate(
    i: int,
    j: int,
    offset: Sequence[int],
) -> tuple[int, int, Tuple[int, int, int]] | None:
    shift = tuple(int(value) for value in offset)
    if i == j:
        if shift == (0, 0, 0):
            return None
        inverse = tuple(-value for value in shift)
        return i, j, min(shift, inverse)
    if i < j:
        return i, j, shift
    return j, i, tuple(-value for value in shift)


def _offset_suffix(offset: Tuple[int, int, int]) -> str:
    if offset == (0, 0, 0):
        return ""
    tokens = [f"m{abs(value)}" if value < 0 else f"p{value}" for value in offset]
    return "_offset_" + "_".join(tokens)


def resolve_bonds(atoms: Atoms, settings: BondSettings) -> BondResolution:
    """按元素对距离与具体原子例外解析当前构型中的化学键。"""
    validate_bond_settings(atoms, settings)
    rules_by_pair = {rule.pair: rule for rule in settings.pair_rules}
    if len(atoms) == 0 or not rules_by_pair:
        return BondResolution((), (), MappingProxyType({}))

    maximum = max(rule.max_distance for rule in settings.pair_rules)
    if maximum <= 0:
        return BondResolution((), (), MappingProxyType({}))

    cutoff = float(np.nextafter(maximum, math.inf))
    first, second, distances, offsets = neighbor_list(
        "ijdS",
        atoms,
        cutoff=cutoff,
        self_interaction=False,
    )
    symbols = atoms.get_chemical_symbols()
    overrides = {
        (override.atom_index, override.pair): override.visibility
        for override in settings.atom_overrides
        if override.visibility is not OverrideVisibility.INHERIT
    }

    candidates: dict[tuple[int, int, Tuple[int, int, int]], float] = {}
    for raw_i, raw_j, raw_distance, raw_offset in zip(
        first, second, distances, offsets
    ):
        canonical = _canonical_candidate(int(raw_i), int(raw_j), raw_offset)
        if canonical is None:
            continue
        previous = candidates.get(canonical)
        distance = float(raw_distance)
        if previous is None or distance < previous:
            candidates[canonical] = distance

    pending = []
    counts: dict[ElementPair, int] = {}
    for (i, j, offset), distance in sorted(candidates.items()):
        pair = normalize_element_pair(symbols[i], symbols[j])
        rule = rules_by_pair.get(pair)
        if rule is None:
            continue
        if not (rule.min_distance <= distance <= rule.max_distance):
            continue

        endpoint_states = (
            overrides.get((i, pair), OverrideVisibility.INHERIT),
            overrides.get((j, pair), OverrideVisibility.INHERIT),
        )
        if OverrideVisibility.HIDE in endpoint_states:
            visible = False
            source = "atom_hide"
        elif OverrideVisibility.SHOW in endpoint_states:
            visible = True
            source = "atom_show"
        else:
            visible = rule.enabled
            source = "pair_enabled" if rule.enabled else "pair_disabled"
        if not settings.draw_bonds:
            visible = False
            source = "global_hidden"

        counts[pair] = counts.get(pair, 0) + 1
        pending.append((i, j, offset, distance, pair, visible, source))

    resolved = []
    for ordinal, (i, j, offset, distance, pair, visible, source) in enumerate(
        pending, start=1
    ):
        base_id = f"bond_{ordinal:04d}_{symbols[i]}{i + 1}_{symbols[j]}{j + 1}"
        resolved.append(
            ResolvedBond(
                i=i,
                j=j,
                offset=offset,
                distance=distance,
                pair=pair,
                bond_id=base_id + _offset_suffix(offset),
                visible=visible,
                visibility_source=source,
            )
        )

    matched = tuple(resolved)
    visible = tuple(bond for bond in matched if bond.visible)
    return BondResolution(matched, visible, MappingProxyType(dict(counts)))


def reapply_bond_visibility(
    resolution: BondResolution,
    settings: BondSettings,
) -> BondResolution:
    """在不重新匹配邻居的前提下，按最终规则更新键显示状态。"""
    rules_by_pair = {rule.pair: rule for rule in settings.pair_rules}
    overrides = {
        (item.atom_index, item.pair): item.visibility
        for item in settings.atom_overrides
        if item.visibility is not OverrideVisibility.INHERIT
    }
    matched = []
    for bond in resolution.matched:
        endpoint_states = (
            overrides.get((bond.i, bond.pair), OverrideVisibility.INHERIT),
            overrides.get((bond.j, bond.pair), OverrideVisibility.INHERIT),
        )
        if OverrideVisibility.HIDE in endpoint_states:
            visible, source = False, "atom_hide"
        elif OverrideVisibility.SHOW in endpoint_states:
            visible, source = True, "atom_show"
        else:
            visible = rules_by_pair[bond.pair].enabled
            source = "pair_enabled" if visible else "pair_disabled"
        if not settings.draw_bonds:
            visible, source = False, "global_hidden"
        matched.append(replace(bond, visible=visible, visibility_source=source))
    matched_tuple = tuple(matched)
    return BondResolution(
        matched_tuple,
        tuple(bond for bond in matched_tuple if bond.visible),
        resolution.match_counts,
    )


def find_override_conflicts(
    atoms: Atoms,
    settings: BondSettings,
) -> Tuple[BondOverrideConflict, ...]:
    """找出已匹配键上的端点例外冲突；最终仍按强制隐藏优先。"""
    resolution = resolve_bonds(atoms, settings)
    overrides = {
        (override.atom_index, override.pair): override.visibility
        for override in settings.atom_overrides
        if override.visibility is not OverrideVisibility.INHERIT
    }
    conflicts = []
    for bond in resolution.matched:
        first = overrides.get((bond.i, bond.pair), OverrideVisibility.INHERIT)
        second = overrides.get((bond.j, bond.pair), OverrideVisibility.INHERIT)
        if {first, second} != {OverrideVisibility.HIDE, OverrideVisibility.SHOW}:
            continue
        hidden = bond.i if first is OverrideVisibility.HIDE else bond.j
        shown = bond.i if first is OverrideVisibility.SHOW else bond.j
        conflicts.append(
            BondOverrideConflict(
                bond_id=bond.bond_id,
                atom_i=bond.i,
                atom_j=bond.j,
                element_pair=bond.pair,
                hidden_atom_index=hidden,
                shown_atom_index=shown,
            )
        )
    return tuple(conflicts)


def count_drawable_bonds(
    atoms: Atoms,
    settings: BondSettings,
    *,
    radius_scale: float,
) -> Mapping[ElementPair, int]:
    """统计规则最终显示且在两个显示球之间仍有外露键段的数量。"""
    scale = _finite_number(radius_scale, "原子半径缩放", "atom radius scale")
    if scale <= 0:
        raise BondRuleError("原子半径缩放必须大于 0")
    resolution = resolve_bonds(atoms, settings)
    counts = {rule.pair: 0 for rule in settings.pair_rules}
    positions = atoms.get_positions()
    radii = covalent_radii[atoms.get_atomic_numbers()] * scale
    cell = np.asarray(atoms.cell)
    for bond in resolution.visible:
        if not is_primary_cell_offset(bond.offset):
            continue
        shifted_j = positions[bond.j] + np.dot(np.asarray(bond.offset), cell)
        segment = clip_bond_to_spheres(
            positions[bond.i],
            shifted_j,
            float(radii[bond.i]),
            float(radii[bond.j]),
        )
        if segment is not None:
            counts[bond.pair] = counts.get(bond.pair, 0) + 1
    return MappingProxyType(counts)
