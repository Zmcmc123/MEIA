"""具体原子的绝对色彩强度、批量选择与草稿状态。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from ase import Atoms
from ase.data import atomic_numbers
from matplotlib.colors import is_color_like, to_hex, to_rgb

from .bond_rules import AtomBondOverride, OverrideVisibility, normalize_element_pair
from .i18n import LocalizedError


def normalize_color_strength(value: object) -> float:
    """校验并规范化 0–1 的绝对色彩强度。"""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LocalizedError(
            "色彩强度必须是 0–1 之间的数值",
            message_key="atom.color_strength_range",
            message_params={"value": repr(value)},
        )
    strength = float(value)
    if not math.isfinite(strength) or not 0.0 <= strength <= 1.0:
        raise LocalizedError(
            "色彩强度必须是 0–1 之间的有限数值",
            message_key="atom.color_strength_range",
            message_params={"value": repr(value)},
        )
    return strength


def apply_color_strength(color: str, strength: object) -> str:
    """从不可变原色向白色混合；重复调用方不会在结果色上累计衰减。"""
    value = normalize_color_strength(strength)
    rgb = to_rgb(color)
    mixed = tuple(value * channel + (1.0 - value) for channel in rgb)
    return to_hex(mixed, keep_alpha=False).upper()


@dataclass(frozen=True, order=True)
class AtomColorStrength:
    """一个按文件顺序定位的具体原子绝对色彩强度。"""

    atom_index: int
    atom_symbol: str
    strength: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.atom_index, bool)
            or not isinstance(self.atom_index, int)
            or self.atom_index < 0
        ):
            raise ValueError("原子索引必须是非负整数")
        if self.atom_symbol not in atomic_numbers:
            raise LocalizedError(
                f"非法元素符号：{self.atom_symbol!r}",
                message_key="atom.invalid_element",
                message_params={"symbol": repr(self.atom_symbol)},
            )
        object.__setattr__(self, "strength", normalize_color_strength(self.strength))


@dataclass(frozen=True, order=True)
class AtomColorOverride:
    """一个按文件顺序定位的具体原子填充基础色。"""

    atom_index: int
    atom_symbol: str
    color: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.atom_index, bool)
            or not isinstance(self.atom_index, int)
            or self.atom_index < 0
        ):
            raise ValueError("原子索引必须是非负整数")
        if self.atom_symbol not in atomic_numbers:
            raise LocalizedError(
                f"非法元素符号：{self.atom_symbol!r}",
                message_key="atom.invalid_element",
                message_params={"symbol": repr(self.atom_symbol)},
            )
        if not isinstance(self.color, str) or not is_color_like(self.color):
            raise LocalizedError(
                f"非法原子颜色：{self.color!r}",
                message_key="atom.invalid_color",
                message_params={"value": repr(self.color)},
            )
        object.__setattr__(self, "color", to_hex(self.color).upper())


def _canonical_color_overrides(
    overrides: Iterable[AtomColorOverride],
) -> tuple[AtomColorOverride, ...]:
    values = tuple(overrides)
    if not all(isinstance(item, AtomColorOverride) for item in values):
        raise TypeError("具体原子颜色必须由 AtomColorOverride 组成")
    by_index: dict[int, AtomColorOverride] = {}
    for item in values:
        if item.atom_index in by_index:
            raise ValueError(f"原子 #{item.atom_index + 1} 的具体颜色重复")
        by_index[item.atom_index] = item
    return tuple(by_index[index] for index in sorted(by_index))


def _canonical_overrides(
    overrides: Iterable[AtomColorStrength],
    default_strength: float = 1.0,
) -> tuple[AtomColorStrength, ...]:
    default = normalize_color_strength(default_strength)
    values = tuple(overrides)
    if not all(isinstance(item, AtomColorStrength) for item in values):
        raise TypeError("具体原子色彩强度必须由 AtomColorStrength 组成")
    by_index: dict[int, AtomColorStrength] = {}
    for item in values:
        if item.atom_index in by_index:
            raise ValueError(f"原子 #{item.atom_index + 1} 的色彩强度重复")
        if item.strength != default:
            by_index[item.atom_index] = item
    return tuple(by_index[index] for index in sorted(by_index))


def validate_atom_color_strengths(
    atoms: Atoms,
    overrides: Sequence[AtomColorStrength],
    default_strength: float = 1.0,
) -> None:
    """验证原子索引与当前构型的元素身份完全一致。"""
    canonical = _canonical_overrides(overrides, default_strength)
    symbols = atoms.get_chemical_symbols()
    for item in canonical:
        if item.atom_index >= len(symbols):
            raise ValueError(f"原子序号 #{item.atom_index + 1} 超出当前构型范围")
        if symbols[item.atom_index] != item.atom_symbol:
            raise ValueError(
                f"原子 #{item.atom_index + 1} 的元素不匹配："
                f"预设为 {item.atom_symbol}，当前为 {symbols[item.atom_index]}"
            )


def color_strength_mapping(
    overrides: Sequence[AtomColorStrength],
    default_strength: float = 1.0,
) -> dict[int, float]:
    """把类型化覆盖转换为渲染配置使用的索引映射。"""
    return {
        item.atom_index: item.strength
        for item in _canonical_overrides(overrides, default_strength)
    }


def atom_color_override_mapping(
    overrides: Sequence[AtomColorOverride],
) -> dict[int, str]:
    """把类型化的具体原子颜色转换为渲染索引映射。"""
    return {item.atom_index: item.color for item in _canonical_color_overrides(overrides)}


def _canonical_bond_overrides(overrides: Iterable[object]) -> tuple[object, ...]:
    values = tuple(overrides)
    if not all(isinstance(item, AtomBondOverride) for item in values):
        raise TypeError("具体原子化学键例外必须由 AtomBondOverride 组成")
    by_key: dict[tuple[int, tuple[str, str]], AtomBondOverride] = {}
    for item in values:
        if item.visibility is OverrideVisibility.INHERIT:
            continue
        key = (item.atom_index, item.pair)
        if key in by_key:
            raise ValueError(
                f"原子 #{item.atom_index + 1} 的 {item.pair[0]}–{item.pair[1]} 键例外重复"
            )
        by_key[key] = item
    return tuple(by_key[key] for key in sorted(by_key))


@dataclass(frozen=True, order=True)
class HiddenAtom:
    """一个按原始文件顺序定位的隐藏原子。"""

    atom_index: int
    atom_symbol: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.atom_index, bool)
            or not isinstance(self.atom_index, int)
            or self.atom_index < 0
        ):
            raise ValueError("原子索引必须是非负整数")
        if self.atom_symbol not in atomic_numbers:
            raise LocalizedError(
                f"非法元素符号：{self.atom_symbol!r}",
                message_key="atom.invalid_element",
                message_params={"symbol": repr(self.atom_symbol)},
            )


@dataclass(frozen=True, order=True)
class AtomHydrogenBondOverride:
    """一个具体原子的氢键可见性例外。"""

    atom_index: int
    atom_symbol: str
    visibility: OverrideVisibility

    def __post_init__(self) -> None:
        if (
            isinstance(self.atom_index, bool)
            or not isinstance(self.atom_index, int)
            or self.atom_index < 0
        ):
            raise ValueError("原子索引必须是非负整数")
        if self.atom_symbol not in atomic_numbers:
            raise ValueError(f"非法元素符号：{self.atom_symbol!r}")
        try:
            visibility = OverrideVisibility(self.visibility)
        except ValueError as exc:
            raise ValueError(f"非法氢键例外状态：{self.visibility!r}") from exc
        object.__setattr__(self, "visibility", visibility)


def _canonical_indexed_records(
    records: Iterable[object],
    record_type: type,
    label: str,
) -> tuple[object, ...]:
    values = tuple(records)
    if not all(isinstance(item, record_type) for item in values):
        raise TypeError(f"{label}必须由 {record_type.__name__} 组成")
    by_index: dict[int, object] = {}
    for item in values:
        atom_index = item.atom_index  # type: ignore[attr-defined]
        if atom_index in by_index:
            raise ValueError(f"原子 #{atom_index + 1} 的{label}重复")
        by_index[atom_index] = item
    return tuple(by_index[index] for index in sorted(by_index))


@dataclass(frozen=True)
class AtomSelectionSettings:
    """当前选区与所有绑定具体原子身份的已应用样式。"""

    selected_atom_indices: tuple[int, ...] = ()
    color_overrides: tuple[AtomColorOverride, ...] = ()
    color_strengths: tuple[AtomColorStrength, ...] = ()
    bond_overrides: tuple[object, ...] = ()
    hidden_atoms: tuple[HiddenAtom, ...] = ()
    hydrogen_bond_overrides: tuple[AtomHydrogenBondOverride, ...] = ()
    default_color_strength: float = 1.0

    def __post_init__(self) -> None:
        selected = tuple(self.selected_atom_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in selected
        ):
            raise ValueError("原子选区只能包含非负整数索引")
        object.__setattr__(self, "selected_atom_indices", tuple(sorted(set(selected))))
        object.__setattr__(
            self, "color_overrides", _canonical_color_overrides(self.color_overrides)
        )
        default_strength = normalize_color_strength(self.default_color_strength)
        object.__setattr__(self, "default_color_strength", default_strength)
        object.__setattr__(
            self,
            "color_strengths",
            _canonical_overrides(self.color_strengths, default_strength),
        )
        object.__setattr__(
            self, "bond_overrides", _canonical_bond_overrides(self.bond_overrides)
        )
        object.__setattr__(
            self,
            "hidden_atoms",
            _canonical_indexed_records(self.hidden_atoms, HiddenAtom, "隐藏记录"),
        )
        object.__setattr__(
            self,
            "hydrogen_bond_overrides",
            _canonical_indexed_records(
                self.hydrogen_bond_overrides,
                AtomHydrogenBondOverride,
                "氢键例外",
            ),
        )


def replace_selected_indices(
    settings: AtomSelectionSettings,
    indices: Iterable[int],
    atom_count: int,
) -> AtomSelectionSettings:
    """只替换规范化选区，保留所有具体原子样式记录。"""
    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count < 0
    ):
        raise ValueError("原子数量必须是非负整数")
    selected = tuple(indices)
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < atom_count
        for index in selected
    ):
        raise LocalizedError(
            "原子选区包含越界原子序号",
            message_key="selection.indices_out_of_range",
            message_params={"count": atom_count},
        )
    return replace(settings, selected_atom_indices=tuple(sorted(set(selected))))


def resolved_color_strengths(
    settings: AtomSelectionSettings,
    atom_count: int,
):
    """展开默认强度和少量例外，供渲染边界使用。"""
    import numpy as np

    if not isinstance(settings, AtomSelectionSettings):
        raise TypeError("原子选择设置必须是 AtomSelectionSettings")
    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count < 0
    ):
        raise ValueError("原子数量必须是非负整数")
    values = np.full(atom_count, settings.default_color_strength, dtype=float)
    for item in settings.color_strengths:
        if item.atom_index >= atom_count:
            raise ValueError(f"原子序号 #{item.atom_index + 1} 超出当前构型范围")
        values[item.atom_index] = item.strength
    return values


def compact_color_strengths(
    symbols: Sequence[str],
    strengths: Sequence[object],
) -> tuple[float, tuple[AtomColorStrength, ...]]:
    """以最常见强度为默认值，将完整序列压缩为例外。"""
    symbol_values = tuple(symbols)
    normalized = tuple(normalize_color_strength(value) for value in strengths)
    if len(symbol_values) != len(normalized):
        raise ValueError("原子元素数量与色彩强度数量不一致")
    if not normalized:
        return 1.0, ()
    counts: dict[float, int] = {}
    for value in normalized:
        counts[value] = counts.get(value, 0) + 1
    default = max(
        counts,
        key=lambda value: (counts[value], value == 1.0, -value),
    )
    exceptions = tuple(
        AtomColorStrength(index, symbol, value)
        for index, (symbol, value) in enumerate(zip(symbol_values, normalized))
        if value != default
    )
    return default, exceptions


@dataclass(frozen=True)
class AtomSelectionOperation:
    """对当前选区执行的一次原子化样式操作。"""

    color_action: str = "unchanged"
    color: str | None = None
    strength: float | None = None
    bond_visibility: Mapping[tuple[str, str], object | None] = MappingProxyType({})
    visibility_action: str = "unchanged"
    hydrogen_bond_visibility: OverrideVisibility | None = None

    def __post_init__(self) -> None:
        if self.color_action not in {"unchanged", "set", "inherit"}:
            raise ValueError(f"非法颜色操作：{self.color_action!r}")
        if self.color_action == "set":
            if not isinstance(self.color, str) or not is_color_like(self.color):
                raise ValueError(f"非法原子颜色：{self.color!r}")
            object.__setattr__(self, "color", to_hex(self.color).upper())
        elif self.color is not None:
            raise ValueError("只有“设置颜色”操作可以携带颜色值")
        if self.strength is not None:
            object.__setattr__(
                self, "strength", normalize_color_strength(self.strength)
            )
        if self.visibility_action not in {"unchanged", "hide", "show"}:
            raise ValueError(f"非法原子可见性操作：{self.visibility_action!r}")
        if self.hydrogen_bond_visibility is not None:
            try:
                hydrogen_visibility = OverrideVisibility(
                    self.hydrogen_bond_visibility
                )
            except ValueError as exc:
                raise ValueError(
                    f"非法氢键例外状态：{self.hydrogen_bond_visibility!r}"
                ) from exc
            object.__setattr__(
                self, "hydrogen_bond_visibility", hydrogen_visibility
            )

        canonical: dict[tuple[str, str], OverrideVisibility | None] = {}
        if not isinstance(self.bond_visibility, Mapping):
            raise TypeError("化学键例外操作必须是元素对映射")
        for raw_pair, raw_visibility in self.bond_visibility.items():
            if not isinstance(raw_pair, (tuple, list)) or len(raw_pair) != 2:
                raise ValueError(f"非法元素对：{raw_pair!r}")
            pair = normalize_element_pair(raw_pair[0], raw_pair[1])
            if pair in canonical:
                raise ValueError(f"化学键操作元素对重复：{pair}")
            if raw_visibility is None:
                visibility = None
            else:
                try:
                    visibility = OverrideVisibility(raw_visibility)
                except ValueError as exc:
                    raise ValueError(
                        f"非法化学键例外状态：{raw_visibility!r}"
                    ) from exc
            canonical[pair] = visibility
        object.__setattr__(self, "bond_visibility", MappingProxyType(canonical))


def _validate_atom_identity(
    symbols: Sequence[str], atom_index: int, atom_symbol: str
) -> None:
    if atom_index >= len(symbols):
        raise LocalizedError(
            f"原子序号 #{atom_index + 1} 超出当前构型范围",
            message_key="selection.index_out_of_range",
            message_params={"index": atom_index + 1, "count": len(symbols)},
        )
    if symbols[atom_index] != atom_symbol:
        raise ValueError(
            f"原子 #{atom_index + 1} 的元素不匹配："
            f"设置为 {atom_symbol}，当前为 {symbols[atom_index]}"
        )


def validate_atom_selection_settings(
    atoms: Atoms,
    settings: AtomSelectionSettings,
    available_pairs: Iterable[tuple[str, str]] | None = None,
) -> None:
    """校验选区与所有具体原子设置属于当前构型。"""
    if not isinstance(settings, AtomSelectionSettings):
        raise TypeError("原子选择设置必须是 AtomSelectionSettings")
    symbols = atoms.get_chemical_symbols()
    for atom_index in settings.selected_atom_indices:
        if atom_index >= len(symbols):
            raise LocalizedError(
                f"原子序号 #{atom_index + 1} 超出当前构型范围",
                message_key="selection.index_out_of_range",
                message_params={"index": atom_index + 1, "count": len(symbols)},
            )
    for item in settings.color_overrides:
        _validate_atom_identity(symbols, item.atom_index, item.atom_symbol)
    validate_atom_color_strengths(
        atoms,
        settings.color_strengths,
        settings.default_color_strength,
    )

    allowed = (
        {normalize_element_pair(*pair) for pair in available_pairs}
        if available_pairs is not None
        else None
    )
    for item in settings.bond_overrides:
        _validate_atom_identity(symbols, item.atom_index, item.atom_symbol)
        if item.atom_symbol not in item.pair:
            raise ValueError(
                f"原子 #{item.atom_index + 1} 不属于元素对 "
                f"{item.pair[0]}–{item.pair[1]}"
            )
        if allowed is not None and item.pair not in allowed:
            raise ValueError(
                f"具体原子例外缺少对应元素对规则：{item.pair}"
            )
    for item in settings.hidden_atoms:
        _validate_atom_identity(symbols, item.atom_index, item.atom_symbol)
    for item in settings.hydrogen_bond_overrides:
        _validate_atom_identity(symbols, item.atom_index, item.atom_symbol)


def apply_atom_selection_operation(
    atoms: Atoms,
    settings: AtomSelectionSettings,
    operation: AtomSelectionOperation,
    available_pairs: Iterable[tuple[str, str]],
) -> AtomSelectionSettings:
    """以原子方式对选区应用颜色、强度和成键可见性。"""
    if not isinstance(operation, AtomSelectionOperation):
        raise TypeError("原子操作必须是 AtomSelectionOperation")
    normalized_pairs = tuple(
        sorted({normalize_element_pair(*pair) for pair in available_pairs})
    )
    validate_atom_selection_settings(atoms, settings, normalized_pairs)
    unknown_pairs = set(operation.bond_visibility) - set(normalized_pairs)
    if unknown_pairs:
        raise ValueError(f"化学键操作缺少对应元素对规则：{sorted(unknown_pairs)}")

    symbols = atoms.get_chemical_symbols()
    colors = {item.atom_index: item for item in settings.color_overrides}
    strengths = {item.atom_index: item for item in settings.color_strengths}
    bonds = {
        (item.atom_index, item.pair): item for item in settings.bond_overrides
    }
    hidden = {item.atom_index: item for item in settings.hidden_atoms}
    hydrogen_bonds = {
        item.atom_index: item for item in settings.hydrogen_bond_overrides
    }

    for atom_index in settings.selected_atom_indices:
        symbol = symbols[atom_index]
        if operation.color_action == "set":
            colors[atom_index] = AtomColorOverride(
                atom_index, symbol, operation.color  # type: ignore[arg-type]
            )
        elif operation.color_action == "inherit":
            colors.pop(atom_index, None)

        if operation.strength is not None:
            if operation.strength == settings.default_color_strength:
                strengths.pop(atom_index, None)
            else:
                strengths[atom_index] = AtomColorStrength(
                    atom_index, symbol, operation.strength
                )

        if operation.visibility_action == "hide":
            hidden[atom_index] = HiddenAtom(atom_index, symbol)
        elif operation.visibility_action == "show":
            hidden.pop(atom_index, None)

        if operation.hydrogen_bond_visibility is not None:
            if operation.hydrogen_bond_visibility is OverrideVisibility.INHERIT:
                hydrogen_bonds.pop(atom_index, None)
            else:
                hydrogen_bonds[atom_index] = AtomHydrogenBondOverride(
                    atom_index,
                    symbol,
                    operation.hydrogen_bond_visibility,
                )

    for pair, visibility in operation.bond_visibility.items():
        if visibility is None:
            continue
        for atom_index in settings.selected_atom_indices:
            symbol = symbols[atom_index]
            if symbol not in pair:
                continue
            key = (atom_index, pair)
            if visibility is OverrideVisibility.INHERIT:
                bonds.pop(key, None)
            else:
                bonds[key] = AtomBondOverride(
                    atom_index,
                    symbol,
                    pair[0],
                    pair[1],
                    visibility,
                )

    candidate = AtomSelectionSettings(
        selected_atom_indices=settings.selected_atom_indices,
        color_overrides=tuple(colors.values()),
        color_strengths=tuple(strengths.values()),
        bond_overrides=tuple(bonds.values()),
        hidden_atoms=tuple(hidden.values()),
        hydrogen_bond_overrides=tuple(hydrogen_bonds.values()),
        default_color_strength=settings.default_color_strength,
    )
    validate_atom_selection_settings(atoms, candidate, normalized_pairs)
    return candidate


@dataclass(frozen=True)
class AtomColorState:
    """与一个构型绑定的已应用/草稿色彩强度和临时批量选择。"""

    structure_id: str
    applied: tuple[AtomColorStrength, ...] = ()
    draft: tuple[AtomColorStrength, ...] = ()
    selected_atom_indices: tuple[int, ...] = ()
    revision: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.structure_id, str) or not self.structure_id.strip():
            raise ValueError("structure_id 必须是非空字符串")
        object.__setattr__(self, "applied", _canonical_overrides(self.applied))
        object.__setattr__(self, "draft", _canonical_overrides(self.draft))
        selected = tuple(self.selected_atom_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in selected
        ):
            raise ValueError("批量选择只能包含非负整数原子索引")
        object.__setattr__(self, "selected_atom_indices", tuple(sorted(set(selected))))
        if isinstance(self.revision, bool) or not isinstance(self.revision, int):
            raise TypeError("revision 必须是整数")
        if self.revision < 0:
            raise ValueError("revision 不能小于 0")

    @property
    def is_dirty(self) -> bool:
        return self.draft != self.applied


def initialize_atom_color_state(
    structure_id: str,
    overrides: Sequence[AtomColorStrength] = (),
) -> AtomColorState:
    canonical = _canonical_overrides(overrides)
    return AtomColorState(structure_id, applied=canonical, draft=canonical)


def set_selected_atom_indices(
    state: AtomColorState,
    atom_indices: Iterable[int],
    atom_count: int,
) -> AtomColorState:
    selected = tuple(sorted(set(atom_indices)))
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < atom_count
        for index in selected
    ):
        raise LocalizedError(
            "批量选择包含越界原子序号",
            message_key="selection.indices_out_of_range",
            message_params={"count": atom_count},
        )
    return replace(state, selected_atom_indices=selected)


def toggle_selected_atom(
    state: AtomColorState,
    atom_index: int,
    atom_count: int,
) -> AtomColorState:
    selected = set(state.selected_atom_indices)
    if atom_index in selected:
        selected.remove(atom_index)
    else:
        selected.add(atom_index)
    return set_selected_atom_indices(state, selected, atom_count)


def invert_selected_atoms(state: AtomColorState, atom_count: int) -> AtomColorState:
    selected = set(range(atom_count)) - set(state.selected_atom_indices)
    return set_selected_atom_indices(state, selected, atom_count)


def set_selected_color_strength(
    state: AtomColorState,
    atoms: Atoms,
    strength: object,
) -> AtomColorState:
    """用绝对目标覆盖所选原子；100% 会删除多余覆盖。"""
    value = normalize_color_strength(strength)
    validate_atom_color_strengths(atoms, state.draft)
    symbols = atoms.get_chemical_symbols()
    by_index = {item.atom_index: item for item in state.draft}
    for index in state.selected_atom_indices:
        if not 0 <= index < len(atoms):
            raise LocalizedError(
                f"原子序号 #{index + 1} 超出当前构型范围",
                message_key="selection.index_out_of_range",
                message_params={"index": index + 1, "count": len(atoms)},
            )
        if value == 1.0:
            by_index.pop(index, None)
        else:
            by_index[index] = AtomColorStrength(index, symbols[index], value)
    return replace(
        state,
        draft=tuple(by_index[index] for index in sorted(by_index)),
    )


def apply_atom_color_draft(state: AtomColorState, atoms: Atoms) -> AtomColorState:
    validate_atom_color_strengths(atoms, state.draft)
    return replace(state, applied=state.draft, revision=state.revision + 1)


def parse_atom_index_expression(text: str, atom_count: int) -> tuple[int, ...]:
    """解析一基原子序号表达式，例如 ``1-3, 5``。"""
    if not isinstance(text, str):
        raise LocalizedError(
            "原子序号表达式必须是文本",
            message_key="selection.expression_text",
            message_params={"value": repr(text)},
        )
    if not text.strip():
        return ()
    indices: set[int] = set()
    for raw_token in text.split(","):
        token = raw_token.strip()
        match = re.fullmatch(r"(\d+)(?:\s*-\s*(\d+))?", token)
        if match is None:
            raise LocalizedError(
                f"原子序号格式无效：{token!r}",
                message_key="selection.invalid_token",
                message_params={"token": repr(token)},
            )
        start = int(match.group(1))
        end = int(match.group(2) or start)
        if start < 1 or end < start or end > atom_count:
            raise LocalizedError(
                f"原子序号范围无效：{token!r}",
                message_key="selection.invalid_range",
                message_params={"token": repr(token), "count": atom_count},
            )
        indices.update(range(start - 1, end))
    return tuple(sorted(indices))


def store_atom_color_state(session_state: Any, state: AtomColorState) -> None:
    session_state["meia_color_structure_id"] = state.structure_id
    session_state["meia_applied_color_strengths"] = state.applied
    session_state["meia_draft_color_strengths"] = state.draft
    session_state["meia_batch_selected_indices"] = state.selected_atom_indices
    session_state["meia_color_revision"] = state.revision


def load_atom_color_state(session_state: Any) -> AtomColorState:
    return AtomColorState(
        structure_id=session_state["meia_color_structure_id"],
        applied=session_state["meia_applied_color_strengths"],
        draft=session_state["meia_draft_color_strengths"],
        selected_atom_indices=session_state["meia_batch_selected_indices"],
        revision=session_state["meia_color_revision"],
    )
