"""严格 v7 通用风格与工作状态快照 JSON。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers, chemical_symbols

from .atom_styles import (
    AtomColorOverride,
    AtomColorStrength,
    AtomHydrogenBondOverride,
    AtomSelectionSettings,
    HiddenAtom,
    validate_atom_selection_settings,
)
from .bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondStrokeStyle,
    normalize_element_pair,
)
from .hydrogen_bonds import HydrogenBondSettings
from .periodic_display import CellPeriodicSettings, PeriodicRange
from .size_profiles import (
    CovalentSizeProfile,
    SizeProfileSettings,
    UniformSizeProfile,
)
from .view_state import CameraState, CameraValidationError
from .visual_state import (
    AtomCellSettings,
    BondModuleSettings,
    ExportSettings,
    PairRuleDefaults,
    PortableStyle,
    ViewSettings,
    VisualizationState,
    apply_portable_style,
)
from .i18n import LocalizedError


SCHEMA_VERSION = 7


class PresetError(LocalizedError):
    """预设无效或与当前严格契约不兼容。"""

    def __init__(
        self,
        technical_message: str,
        *,
        message_key: str,
        message_params: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(
            technical_message,
            message_key=message_key,
            message_params=message_params,
        )


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PresetError(
            f"{label}必须是数值",
            message_key="preset.number_required",
            message_params={"field": label, "value": repr(value)},
        )
    result = float(value)
    if not math.isfinite(result):
        raise PresetError(
            f"{label}必须是有限数值",
            message_key="preset.finite_required",
            message_params={"field": label, "value": repr(value)},
        )
    return result


def _positive_number(value: object, label: str) -> float:
    result = _number(value, label)
    if result <= 0:
        raise PresetError(
            f"{label}必须大于 0",
            message_key="preset.positive_required",
            message_params={"field": label, "value": repr(value)},
        )
    return result


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PresetError(
            f"{label}必须是整数",
            message_key="preset.integer_required",
            message_params={"field": label, "value": repr(value)},
        )
    return value


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise PresetError(
            f"{label}必须是布尔值",
            message_key="preset.boolean_required",
            message_params={"field": label, "value": repr(value)},
        )
    return value


def _positive_number_mapping(value: object, label: str) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise PresetError(
            f"{label} 必须是 JSON 对象",
            message_key="preset.object_required",
            message_params={"field": label},
        )
    return {
        key: _positive_number(item, f"{label}.{key}")
        for key, item in value.items()
    }


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PresetError(
            f"{label}必须是非空字符串",
            message_key="preset.string_required",
            message_params={"field": label, "value": repr(value)},
        )
    return value.strip()


def _object(
    value: object,
    label: str,
    required: set[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PresetError(
            f"{label} 必须是 JSON 对象",
            message_key="preset.object_required",
            message_params={"field": label},
        )
    missing = sorted(required - set(value))
    if missing:
        raise PresetError(
            f"{label} 缺少字段：{', '.join(missing)}",
            message_key="preset.missing_fields",
            message_params={"field": label, "fields": ", ".join(missing)},
        )
    unknown = sorted(set(value) - required)
    if unknown:
        raise PresetError(
            f"{label} 包含未知字段：{', '.join(unknown)}",
            message_key="preset.unknown_fields",
            message_params={"field": label, "fields": ", ".join(unknown)},
        )
    return value


def _array(value: object, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise PresetError(
            f"{label} 必须是数组",
            message_key="preset.array_required",
            message_params={"field": label},
        )
    return value


def _two_elements(value: object, label: str) -> tuple[str, str]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(item, str) for item in value)
    ):
        raise PresetError(
            f"{label} 必须是两个元素符号的数组",
            message_key="preset.element_pair_required",
            message_params={"field": label},
        )
    return value[0], value[1]


class PresetKind(str, Enum):
    """v7 JSON 的两种用户可见预设类型。"""

    STYLE = "style"
    WORKSPACE_SNAPSHOT = "workspace_snapshot"


@dataclass(frozen=True)
class PresetMetadata:
    """v7 通用根元数据。"""

    schema_version: int
    preset_kind: PresetKind
    name: str
    created_at: str
    meia_version: str

    def __post_init__(self) -> None:
        version = _integer(self.schema_version, "schema_version")
        if version != SCHEMA_VERSION:
            raise PresetError(
                f"v7 预设的 schema_version 必须是 {SCHEMA_VERSION}",
                message_key="preset.unsupported_schema",
                message_params={
                    "version": repr(version),
                    "supported": SCHEMA_VERSION,
                },
            )
        try:
            kind = PresetKind(self.preset_kind)
        except (TypeError, ValueError) as exc:
            raise PresetError(
                f"非法预设类型：{self.preset_kind!r}",
                message_key="preset.invalid_kind",
                message_params={"value": repr(self.preset_kind)},
            ) from exc
        name = _nonempty_string(self.name, "name")
        created_at = _nonempty_string(self.created_at, "created_at")
        try:
            datetime.fromisoformat(created_at)
        except ValueError as exc:
            raise PresetError(
                "创建时间必须是 ISO 8601 格式",
                message_key="preset.created_at_iso",
                message_params={"value": repr(created_at)},
            ) from exc
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "preset_kind", kind)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(
            self,
            "meia_version",
            _nonempty_string(self.meia_version, "meia_version"),
        )


def _validate_complete_palette(style: PortableStyle) -> None:
    actual = set(style.atom_cell.element_colors)
    expected = set(chemical_symbols[1:119])
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        details = []
        if missing:
            details.append(f"缺少元素 {', '.join(missing)}")
        if extra:
            details.append(f"非法元素 {', '.join(extra)}")
        raise PresetError(
            "元素配色必须完整覆盖周期表：" + "；".join(details),
            message_key="preset.incomplete_palette",
            message_params={
                "missing": ", ".join(missing) or "none",
                "extra": ", ".join(extra) or "none",
            },
        )


@dataclass(frozen=True)
class SnapshotStructure:
    """工作状态快照内嵌的规范化 ASE 结构。"""

    source_name: str
    symbols: tuple[str, ...]
    positions_angstrom: tuple[tuple[float, float, float], ...]
    cell_angstrom: tuple[tuple[float, float, float], ...]
    pbc: tuple[bool, bool, bool]

    def __post_init__(self) -> None:
        source_name = _nonempty_string(self.source_name, "structure.source_name")
        if not isinstance(self.symbols, (tuple, list)):
            raise PresetError(
                "structure.symbols 必须是元素符号数组",
                message_key="preset.symbols_required",
            )
        symbols = tuple(self.symbols)
        for symbol in symbols:
            if not isinstance(symbol, str) or symbol not in atomic_numbers:
                raise PresetError(
                    f"structure.symbols 包含非法元素：{symbol!r}",
                    message_key="preset.invalid_symbol",
                    message_params={"field": "structure.symbols", "value": repr(symbol)},
                )

        try:
            positions = np.asarray(self.positions_angstrom, dtype=float)
        except (TypeError, ValueError) as exc:
            raise PresetError(
                "structure.positions_angstrom 必须是数值矩阵",
                message_key="preset.numeric_matrix_required",
                message_params={"field": "structure.positions_angstrom"},
            ) from exc
        if positions.shape != (len(symbols), 3):
            raise PresetError(
                "structure.positions_angstrom 行数必须与元素数相同，"
                "且每行必须有 3 个坐标",
                message_key="preset.positions_shape",
                message_params={"count": len(symbols), "shape": repr(positions.shape)},
            )
        if not np.isfinite(positions).all():
            raise PresetError(
                "structure.positions_angstrom 必须只包含有限数值",
                message_key="preset.finite_matrix_required",
                message_params={"field": "structure.positions_angstrom"},
            )

        try:
            cell = np.asarray(self.cell_angstrom, dtype=float)
        except (TypeError, ValueError) as exc:
            raise PresetError(
                "structure.cell_angstrom 必须是数值矩阵",
                message_key="preset.numeric_matrix_required",
                message_params={"field": "structure.cell_angstrom"},
            ) from exc
        if cell.shape != (3, 3):
            raise PresetError(
                "structure.cell_angstrom 必须是 3×3 矩阵",
                message_key="preset.matrix_shape",
                message_params={
                    "field": "structure.cell_angstrom",
                    "expected": "(3, 3)",
                    "shape": repr(cell.shape),
                },
            )
        if not np.isfinite(cell).all():
            raise PresetError(
                "structure.cell_angstrom 必须只包含有限数值",
                message_key="preset.finite_matrix_required",
                message_params={"field": "structure.cell_angstrom"},
            )

        if (
            not isinstance(self.pbc, (tuple, list))
            or len(self.pbc) != 3
            or not all(isinstance(value, bool) for value in self.pbc)
        ):
            raise PresetError(
                "structure.PBC 必须是三个布尔值",
                message_key="preset.pbc_required",
            )

        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(
            self,
            "positions_angstrom",
            tuple(tuple(float(value) for value in row) for row in positions),
        )
        object.__setattr__(
            self,
            "cell_angstrom",
            tuple(tuple(float(value) for value in row) for row in cell),
        )
        object.__setattr__(self, "pbc", tuple(self.pbc))

    @classmethod
    def from_atoms(cls, atoms: Atoms, source_name: str) -> "SnapshotStructure":
        if not isinstance(atoms, Atoms):
            raise PresetError(
                "快照结构必须来自 ASE Atoms",
                message_key="preset.type_required",
                message_params={"field": "structure", "expected": "ASE Atoms"},
            )
        return cls(
            source_name=source_name,
            symbols=tuple(atoms.get_chemical_symbols()),
            positions_angstrom=tuple(
                tuple(float(value) for value in row) for row in atoms.positions
            ),
            cell_angstrom=tuple(
                tuple(float(value) for value in row) for row in atoms.cell.array
            ),
            pbc=tuple(bool(value) for value in atoms.pbc),
        )

    def to_atoms(self) -> Atoms:
        return Atoms(
            symbols=list(self.symbols),
            positions=np.asarray(self.positions_angstrom, dtype=float),
            cell=np.asarray(self.cell_angstrom, dtype=float),
            pbc=list(self.pbc),
        )


@dataclass(frozen=True)
class StylePreset:
    """不包含结构或具体原子设置的 v7 通用风格。"""

    metadata: PresetMetadata
    style: PortableStyle

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PresetMetadata):
            raise PresetError(
                "通用风格缺少 PresetMetadata",
                message_key="preset.type_required",
                message_params={"field": "metadata", "expected": "PresetMetadata"},
            )
        if self.metadata.preset_kind is not PresetKind.STYLE:
            raise PresetError(
                "通用风格的 preset_kind 必须是 style",
                message_key="preset.expected_kind",
                message_params={
                    "expected": PresetKind.STYLE.value,
                    "actual": self.metadata.preset_kind.value,
                },
            )
        if not isinstance(self.style, PortableStyle):
            raise PresetError(
                "通用风格必须包含 PortableStyle",
                message_key="preset.type_required",
                message_params={"field": "style", "expected": "PortableStyle"},
            )
        _validate_complete_palette(self.style)


@dataclass(frozen=True)
class WorkspaceSnapshot:
    """包含规范化结构和具体原子状态的 v7 工作状态快照。"""

    metadata: PresetMetadata
    structure: SnapshotStructure
    state: VisualizationState

    def __post_init__(self) -> None:
        if not isinstance(self.metadata, PresetMetadata):
            raise PresetError(
                "工作状态快照缺少 PresetMetadata",
                message_key="preset.type_required",
                message_params={"field": "metadata", "expected": "PresetMetadata"},
            )
        if self.metadata.preset_kind is not PresetKind.WORKSPACE_SNAPSHOT:
            raise PresetError(
                "工作状态快照的 preset_kind 必须是 workspace_snapshot",
                message_key="preset.expected_kind",
                message_params={
                    "expected": PresetKind.WORKSPACE_SNAPSHOT.value,
                    "actual": self.metadata.preset_kind.value,
                },
            )
        if not isinstance(self.structure, SnapshotStructure):
            raise PresetError(
                "工作状态快照缺少 SnapshotStructure",
                message_key="preset.type_required",
                message_params={"field": "structure", "expected": "SnapshotStructure"},
            )
        if not isinstance(self.state, VisualizationState):
            raise PresetError(
                "工作状态快照缺少 VisualizationState",
                message_key="preset.type_required",
                message_params={"field": "state", "expected": "VisualizationState"},
            )
        _validate_complete_palette(self.state.style)
        atoms = self.structure.to_atoms()
        pairs = tuple(rule.pair for rule in self.state.style.bonds.pair_rules)
        try:
            validate_atom_selection_settings(atoms, self.state.atom_selection, pairs)
        except LocalizedError as exc:
            raise PresetError(
                str(exc),
                message_key=exc.message_key or "preset.invalid_value",
                message_params=exc.message_params,
            ) from exc
        except (TypeError, ValueError) as exc:
            raise PresetError(
                str(exc),
                message_key="preset.invalid_value_type",
                message_params={"error_type": type(exc).__name__},
            ) from exc


def _metadata_mapping(metadata: PresetMetadata) -> dict[str, Any]:
    return {
        "schema_version": metadata.schema_version,
        "preset_kind": metadata.preset_kind.value,
        "name": metadata.name,
        "created_at": metadata.created_at,
        "meia_version": metadata.meia_version,
    }


def _ordered_element_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    """按原子序数编码元素映射，保证跨会话字节稳定。"""
    return {
        symbol: values[symbol]
        for symbol in sorted(values, key=atomic_numbers.__getitem__)
    }


def _style_sections(style: PortableStyle) -> dict[str, Any]:
    return {
        "view": {
            "rotation": style.view.rotation,
            "camera": style.view.camera.to_plotly_dict(),
        },
        "size_profiles": {
            "active_mode": style.size_profiles.active_mode.value,
            "covalent": {
                "global_scale": style.size_profiles.covalent.global_scale,
                "reference_overrides_angstrom": _ordered_element_mapping(
                    style.size_profiles.covalent.reference_overrides_angstrom
                ),
                "bond_width_ratio": (
                    style.size_profiles.covalent.bond_width_ratio
                ),
            },
            "uniform": {
                "global_scale": style.size_profiles.uniform.global_scale,
                "reference_radius_angstrom": (
                    style.size_profiles.uniform.reference_radius_angstrom
                ),
                "reference_overrides_angstrom": _ordered_element_mapping(
                    style.size_profiles.uniform.reference_overrides_angstrom
                ),
                "bond_width_ratio": style.size_profiles.uniform.bond_width_ratio,
            },
        },
        "atoms": {
            "outline_width": style.atom_cell.outline_width,
            "element_colors": _ordered_element_mapping(
                style.atom_cell.element_colors
            ),
        },
        "bonds": {
            "draw_bonds": style.bonds.draw_bonds,
            "style": {
                "stroke_width": style.bonds.style.stroke_width,
                "stroke_color": style.bonds.style.stroke_color,
            },
            "pair_rule_defaults": {
                "bond_cutoff": style.bonds.defaults.bond_cutoff,
                "long_distance_threshold_angstrom": (
                    style.bonds.defaults.long_distance_threshold_angstrom
                ),
                "pair_distance_multipliers": [
                    {"elements": [a, b], "multiplier": multiplier}
                    for a, b, multiplier in style.bonds.defaults.pair_distance_multipliers
                ],
            },
            "hydrogen_bonds": {
                "draw": style.bonds.hydrogen_bonds.draw,
                "max_hydrogen_oxygen_distance_angstrom": (
                    style.bonds.hydrogen_bonds.max_hydrogen_oxygen_distance
                ),
                "min_angle_degrees": (
                    style.bonds.hydrogen_bonds.min_angle_degrees
                ),
            },
            "pair_rules": [
                {
                    "elements": list(rule.pair),
                    "enabled": rule.enabled,
                    "participates_in_periodic_unwrap": (
                        rule.participates_in_periodic_unwrap
                    ),
                    "min_distance_angstrom": rule.min_distance,
                    "max_distance_angstrom": rule.max_distance,
                }
                for rule in style.bonds.pair_rules
            ],
        },
        "cell_periodic": {
            "show_unit_cell": style.cell_periodic.show_unit_cell,
            "unwrap_bonded_groups": style.cell_periodic.unwrap_bonded_groups,
            "ranges": {
                axis: {"start": value.start, "end": value.end}
                for axis, value in (
                    ("a", style.cell_periodic.a),
                    ("b", style.cell_periodic.b),
                    ("c", style.cell_periodic.c),
                )
            },
        },
        "export": {
            "format": style.export.format,
            "dpi": style.export.dpi,
            "transparent": style.export.transparent,
        },
    }


def _atom_selection_mapping(settings: AtomSelectionSettings) -> dict[str, Any]:
    return {
        "selected_indices": list(settings.selected_atom_indices),
        "color_overrides": [
            {
                "atom_index": item.atom_index,
                "atom_symbol": item.atom_symbol,
                "color": item.color,
            }
            for item in settings.color_overrides
        ],
        "color_strengths": [
            {
                "atom_index": item.atom_index,
                "atom_symbol": item.atom_symbol,
                "strength": item.strength,
            }
            for item in settings.color_strengths
        ],
        "bond_overrides": [
            {
                "atom_index": item.atom_index,
                "atom_symbol": item.atom_symbol,
                "elements": list(item.pair),
                "visibility": item.visibility.value,
            }
            for item in settings.bond_overrides
        ],
        "hidden_atoms": [
            {"atom_index": item.atom_index, "atom_symbol": item.atom_symbol}
            for item in settings.hidden_atoms
        ],
        "hydrogen_bond_overrides": [
            {
                "atom_index": item.atom_index,
                "atom_symbol": item.atom_symbol,
                "visibility": item.visibility.value,
            }
            for item in settings.hydrogen_bond_overrides
        ],
    }


def _encode_preset(root: Mapping[str, Any]) -> str:
    try:
        return json.dumps(
            root,
            ensure_ascii=False,
            sort_keys=False,
            indent=2,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise PresetError(
            f"预设无法编码为 JSON：{exc}",
            message_key="preset.encode_failed",
            message_params={
                "error_type": type(exc).__name__,
                "detail": str(exc),
            },
        ) from exc


def style_preset_to_json(preset: StylePreset) -> str:
    """编码不带任何结构或原子索引的 v7 通用风格。"""
    if not isinstance(preset, StylePreset):
        raise PresetError(
            "只能用此接口导出 StylePreset",
            message_key="preset.type_required",
            message_params={"field": "preset", "expected": "StylePreset"},
        )
    root = _metadata_mapping(preset.metadata)
    root.update(_style_sections(preset.style))
    return _encode_preset(root)


def workspace_snapshot_to_json(snapshot: WorkspaceSnapshot) -> str:
    """编码可精确恢复内存结构的 v7 工作状态快照。"""
    if not isinstance(snapshot, WorkspaceSnapshot):
        raise PresetError(
            "只能用此接口导出 WorkspaceSnapshot",
            message_key="preset.type_required",
            message_params={"field": "snapshot", "expected": "WorkspaceSnapshot"},
        )
    root = _metadata_mapping(snapshot.metadata)
    root.update(_style_sections(snapshot.state.style))
    root["structure"] = {
        "source_name": snapshot.structure.source_name,
        "symbols": list(snapshot.structure.symbols),
        "positions_angstrom": [
            list(row) for row in snapshot.structure.positions_angstrom
        ],
        "cell_angstrom": [list(row) for row in snapshot.structure.cell_angstrom],
        "pbc": list(snapshot.structure.pbc),
    }
    root["atom_selection"] = _atom_selection_mapping(snapshot.state.atom_selection)
    return _encode_preset(root)


def _parse_camera(value: object) -> CameraState:
    camera = _object(value, "view.camera", {"eye", "up", "center", "projection"})
    for group_name in ("eye", "up", "center"):
        group = _object(
            camera[group_name], f"view.camera.{group_name}", {"x", "y", "z"}
        )
        for axis in ("x", "y", "z"):
            _number(group[axis], f"view.camera.{group_name}.{axis}")
    _object(camera["projection"], "view.camera.projection", {"type"})
    try:
        return CameraState.from_mapping(camera)
    except CameraValidationError as exc:
        raise PresetError(
            f"相机参数无效：{exc}",
            message_key=exc.message_key,
            message_params=exc.message_params,
        ) from exc


def _parse_periodic_range(value: object, axis: str) -> PeriodicRange:
    range_value = _object(
        value,
        f"cell_periodic.ranges.{axis}",
        {"start", "end"},
    )
    return PeriodicRange(
        _integer(range_value["start"], f"cell_periodic.ranges.{axis}.start"),
        _integer(range_value["end"], f"cell_periodic.ranges.{axis}.end"),
    )


def _parse_style(root: Mapping[str, Any]) -> PortableStyle:
    view = _object(root["view"], "view", {"rotation", "camera"})
    size_profiles = _object(
        root["size_profiles"],
        "size_profiles",
        {"active_mode", "covalent", "uniform"},
    )
    covalent_profile = _object(
        size_profiles["covalent"],
        "size_profiles.covalent",
        {"global_scale", "reference_overrides_angstrom", "bond_width_ratio"},
    )
    uniform_profile = _object(
        size_profiles["uniform"],
        "size_profiles.uniform",
        {
            "global_scale",
            "reference_radius_angstrom",
            "reference_overrides_angstrom",
            "bond_width_ratio",
        },
    )
    atoms = _object(
        root["atoms"],
        "atoms",
        {"outline_width", "element_colors"},
    )
    cell_periodic = _object(
        root["cell_periodic"],
        "cell_periodic",
        {"show_unit_cell", "unwrap_bonded_groups", "ranges"},
    )
    ranges = _object(
        cell_periodic["ranges"],
        "cell_periodic.ranges",
        {"a", "b", "c"},
    )
    bonds = _object(
        root["bonds"],
        "bonds",
        {
            "draw_bonds",
            "style",
            "pair_rule_defaults",
            "pair_rules",
            "hydrogen_bonds",
        },
    )
    bond_style = _object(
        bonds["style"],
        "bonds.style",
        {"stroke_width", "stroke_color"},
    )
    defaults = _object(
        bonds["pair_rule_defaults"],
        "bonds.pair_rule_defaults",
        {
            "bond_cutoff",
            "long_distance_threshold_angstrom",
            "pair_distance_multipliers",
        },
    )
    hydrogen_bonds = _object(
        bonds["hydrogen_bonds"],
        "bonds.hydrogen_bonds",
        {
            "draw",
            "max_hydrogen_oxygen_distance_angstrom",
            "min_angle_degrees",
        },
    )
    export = _object(root["export"], "export", {"format", "dpi", "transparent"})

    if not isinstance(atoms["element_colors"], Mapping):
        raise PresetError(
            "atoms.element_colors 必须是 JSON 对象",
            message_key="preset.object_required",
            message_params={"field": "atoms.element_colors"},
        )
    multipliers = []
    multiplier_pairs: set[tuple[str, str]] = set()
    for index, raw_item in enumerate(
        _array(
            defaults["pair_distance_multipliers"],
            "bonds.pair_rule_defaults.pair_distance_multipliers",
        )
    ):
        item = _object(
            raw_item,
            f"bonds.pair_rule_defaults.pair_distance_multipliers[{index}]",
            {"elements", "multiplier"},
        )
        element_a, element_b = _two_elements(
            item["elements"],
            f"bonds.pair_rule_defaults.pair_distance_multipliers[{index}].elements",
        )
        multiplier_field = (
            f"bonds.pair_rule_defaults.pair_distance_multipliers[{index}].multiplier"
        )
        multiplier = _positive_number(item["multiplier"], multiplier_field)
        pair = normalize_element_pair(element_a, element_b)
        if pair in multiplier_pairs:
            raise PresetError(
                f"元素对距离乘数重复：{pair}",
                message_key="preset.duplicate_pair_multiplier",
                message_params={"pair": "–".join(pair)},
            )
        multiplier_pairs.add(pair)
        multipliers.append((pair[0], pair[1], multiplier))

    pair_rules = []
    for index, raw_rule in enumerate(
        _array(bonds["pair_rules"], "bonds.pair_rules")
    ):
        rule = _object(
            raw_rule,
            f"bonds.pair_rules[{index}]",
            {
                "elements",
                "enabled",
                "participates_in_periodic_unwrap",
                "min_distance_angstrom",
                "max_distance_angstrom",
            },
        )
        element_a, element_b = _two_elements(
            rule["elements"], f"bonds.pair_rules[{index}].elements"
        )
        pair_rules.append(
            BondPairRule(
                element_a,
                element_b,
                _number(
                    rule["min_distance_angstrom"],
                    f"bonds.pair_rules[{index}].min_distance_angstrom",
                ),
                _number(
                    rule["max_distance_angstrom"],
                    f"bonds.pair_rules[{index}].max_distance_angstrom",
                ),
                enabled=_boolean(
                    rule["enabled"], f"bonds.pair_rules[{index}].enabled"
                ),
                participates_in_periodic_unwrap=_boolean(
                    rule["participates_in_periodic_unwrap"],
                    f"bonds.pair_rules[{index}].participates_in_periodic_unwrap",
                ),
            )
        )

    rotation = _nonempty_string(view["rotation"], "view.rotation")
    try:
        parsed_view = ViewSettings(rotation, _parse_camera(view["camera"]))
    except PresetError:
        raise
    except (TypeError, ValueError) as exc:
        raise PresetError(
            f"非法视角旋转：{rotation!r}",
            message_key="preset.invalid_rotation",
            message_params={"value": repr(rotation)},
        ) from exc

    parsed_size_profiles = SizeProfileSettings(
        active_mode=size_profiles["active_mode"],
        covalent=CovalentSizeProfile(
            global_scale=_positive_number(
                covalent_profile["global_scale"],
                "size_profiles.covalent.global_scale",
            ),
            reference_overrides_angstrom=_positive_number_mapping(
                covalent_profile["reference_overrides_angstrom"],
                "size_profiles.covalent.reference_overrides_angstrom",
            ),
            bond_width_ratio=_positive_number(
                covalent_profile["bond_width_ratio"],
                "size_profiles.covalent.bond_width_ratio",
            ),
        ),
        uniform=UniformSizeProfile(
            global_scale=_positive_number(
                uniform_profile["global_scale"],
                "size_profiles.uniform.global_scale",
            ),
            reference_radius_angstrom=_positive_number(
                uniform_profile["reference_radius_angstrom"],
                "size_profiles.uniform.reference_radius_angstrom",
            ),
            reference_overrides_angstrom=_positive_number_mapping(
                uniform_profile["reference_overrides_angstrom"],
                "size_profiles.uniform.reference_overrides_angstrom",
            ),
            bond_width_ratio=_positive_number(
                uniform_profile["bond_width_ratio"],
                "size_profiles.uniform.bond_width_ratio",
            ),
        ),
    )

    export_format = _nonempty_string(export["format"], "export.format").lower()
    if export_format not in {"svg", "png", "pdf"}:
        raise PresetError(
            f"不支持的导出格式：{export_format!r}",
            message_key="preset.invalid_export_format",
            message_params={"value": repr(export_format)},
        )
    export_dpi = _integer(export["dpi"], "export.dpi")
    if export_dpi <= 0:
        raise PresetError(
            "export.dpi 必须大于 0",
            message_key="preset.positive_integer_required",
            message_params={"field": "export.dpi", "value": repr(export["dpi"])},
        )

    return PortableStyle(
        view=parsed_view,
        size_profiles=parsed_size_profiles,
        atom_cell=AtomCellSettings(
            outline_width=_number(atoms["outline_width"], "atoms.outline_width"),
            element_colors=atoms["element_colors"],
        ),
        bonds=BondModuleSettings(
            draw_bonds=_boolean(bonds["draw_bonds"], "bonds.draw_bonds"),
            style=BondStrokeStyle(
                _number(bond_style["stroke_width"], "bonds.style.stroke_width"),
                bond_style["stroke_color"],
            ),
            defaults=PairRuleDefaults(
                bond_cutoff=_positive_number(
                    defaults["bond_cutoff"],
                    "bonds.pair_rule_defaults.bond_cutoff",
                ),
                long_distance_threshold_angstrom=_positive_number(
                    defaults["long_distance_threshold_angstrom"],
                    "bonds.pair_rule_defaults.long_distance_threshold_angstrom",
                ),
                pair_distance_multipliers=tuple(multipliers),
            ),
            pair_rules=tuple(pair_rules),
            hydrogen_bonds=HydrogenBondSettings(
                _boolean(hydrogen_bonds["draw"], "bonds.hydrogen_bonds.draw"),
                _positive_number(
                    hydrogen_bonds["max_hydrogen_oxygen_distance_angstrom"],
                    "bonds.hydrogen_bonds.max_hydrogen_oxygen_distance_angstrom",
                ),
                _number(
                    hydrogen_bonds["min_angle_degrees"],
                    "bonds.hydrogen_bonds.min_angle_degrees",
                ),
            ),
        ),
        cell_periodic=CellPeriodicSettings(
            show_unit_cell=_integer(
                cell_periodic["show_unit_cell"], "cell_periodic.show_unit_cell"
            ),
            unwrap_bonded_groups=_boolean(
                cell_periodic["unwrap_bonded_groups"],
                "cell_periodic.unwrap_bonded_groups",
            ),
            a=_parse_periodic_range(ranges["a"], "a"),
            b=_parse_periodic_range(ranges["b"], "b"),
            c=_parse_periodic_range(ranges["c"], "c"),
        ),
        export=ExportSettings(
            export_format,
            export_dpi,
            _boolean(export["transparent"], "export.transparent"),
        ),
    )


def _parse_metadata(root: Mapping[str, Any]) -> PresetMetadata:
    return PresetMetadata(
        root["schema_version"],
        root["preset_kind"],
        root["name"],
        root["created_at"],
        root["meia_version"],
    )


def _parse_structure(value: object) -> SnapshotStructure:
    structure = _object(
        value,
        "structure",
        {"source_name", "symbols", "positions_angstrom", "cell_angstrom", "pbc"},
    )
    return SnapshotStructure(
        structure["source_name"],
        structure["symbols"],
        structure["positions_angstrom"],
        structure["cell_angstrom"],
        structure["pbc"],
    )


def _parse_atom_selection(value: object) -> AtomSelectionSettings:
    selection = _object(
        value,
        "atom_selection",
        {
            "selected_indices",
            "color_overrides",
            "color_strengths",
            "bond_overrides",
            "hidden_atoms",
            "hydrogen_bond_overrides",
        },
    )
    selected = _array(selection["selected_indices"], "atom_selection.selected_indices")
    colors = []
    for index, raw_item in enumerate(
        _array(selection["color_overrides"], "atom_selection.color_overrides")
    ):
        item = _object(
            raw_item,
            f"atom_selection.color_overrides[{index}]",
            {"atom_index", "atom_symbol", "color"},
        )
        colors.append(
            AtomColorOverride(item["atom_index"], item["atom_symbol"], item["color"])
        )
    strengths = []
    for index, raw_item in enumerate(
        _array(selection["color_strengths"], "atom_selection.color_strengths")
    ):
        item = _object(
            raw_item,
            f"atom_selection.color_strengths[{index}]",
            {"atom_index", "atom_symbol", "strength"},
        )
        strengths.append(
            AtomColorStrength(
                item["atom_index"], item["atom_symbol"], item["strength"]
            )
        )
    bond_overrides = []
    for index, raw_item in enumerate(
        _array(selection["bond_overrides"], "atom_selection.bond_overrides")
    ):
        item = _object(
            raw_item,
            f"atom_selection.bond_overrides[{index}]",
            {"atom_index", "atom_symbol", "elements", "visibility"},
        )
        element_a, element_b = _two_elements(
            item["elements"], f"atom_selection.bond_overrides[{index}].elements"
        )
        bond_overrides.append(
            AtomBondOverride(
                item["atom_index"],
                item["atom_symbol"],
                element_a,
                element_b,
                item["visibility"],
            )
        )
    hidden_atoms = []
    for index, raw_item in enumerate(
        _array(selection["hidden_atoms"], "atom_selection.hidden_atoms")
    ):
        item = _object(
            raw_item,
            f"atom_selection.hidden_atoms[{index}]",
            {"atom_index", "atom_symbol"},
        )
        hidden_atoms.append(HiddenAtom(item["atom_index"], item["atom_symbol"]))
    hydrogen_bond_overrides = []
    for index, raw_item in enumerate(
        _array(
            selection["hydrogen_bond_overrides"],
            "atom_selection.hydrogen_bond_overrides",
        )
    ):
        item = _object(
            raw_item,
            f"atom_selection.hydrogen_bond_overrides[{index}]",
            {"atom_index", "atom_symbol", "visibility"},
        )
        hydrogen_bond_overrides.append(
            AtomHydrogenBondOverride(
                item["atom_index"], item["atom_symbol"], item["visibility"]
            )
        )
    return AtomSelectionSettings(
        selected_atom_indices=tuple(selected),
        color_overrides=tuple(colors),
        color_strengths=tuple(strengths),
        bond_overrides=tuple(bond_overrides),
        hidden_atoms=tuple(hidden_atoms),
        hydrogen_bond_overrides=tuple(hydrogen_bond_overrides),
    )


def _reject_duplicate_keys(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PresetError(
                f"JSON 对象包含重复字段：{key}",
                message_key="preset.duplicate_field",
                message_params={"field": key},
            )
        result[key] = value
    return result


def parse_preset(payload: str | bytes) -> StylePreset | WorkspaceSnapshot:
    """严格解析 v7 通用风格或工作状态快照。"""
    try:
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        if not isinstance(payload, str):
            raise TypeError("payload must be str or bytes")
        root_value = json.loads(payload, object_pairs_hook=_reject_duplicate_keys)
    except PresetError:
        raise
    except (UnicodeDecodeError, TypeError, json.JSONDecodeError) as exc:
        raise PresetError(
            f"预设 JSON 无效：{exc}",
            message_key="preset.invalid_json",
            message_params={"detail": str(exc)},
        ) from exc
    if not isinstance(root_value, Mapping):
        raise PresetError(
            "预设必须是 JSON 对象",
            message_key="preset.root_object_required",
        )

    version = root_value.get("schema_version")
    if version != SCHEMA_VERSION:
        raise PresetError(
            f"不支持的预设格式版本：{version!r}；仅支持 v7",
            message_key="preset.unsupported_schema",
            message_params={
                "version": repr(version),
                "supported": SCHEMA_VERSION,
            },
        )
    try:
        kind = PresetKind(root_value.get("preset_kind"))
    except (TypeError, ValueError) as exc:
        value = root_value.get("preset_kind")
        raise PresetError(
            f"非法预设类型：{value!r}",
            message_key="preset.invalid_kind",
            message_params={"value": repr(value)},
        ) from exc

    common = {
        "schema_version",
        "preset_kind",
        "name",
        "created_at",
        "meia_version",
        "view",
        "size_profiles",
        "atoms",
        "bonds",
        "cell_periodic",
        "export",
    }
    try:
        if kind is PresetKind.STYLE:
            root = _object(root_value, "preset", common)
            return StylePreset(_parse_metadata(root), _parse_style(root))
        root = _object(
            root_value,
            "preset",
            common | {"structure", "atom_selection"},
        )
        style = _parse_style(root)
        return WorkspaceSnapshot(
            metadata=_parse_metadata(root),
            structure=_parse_structure(root["structure"]),
            state=VisualizationState(
                style=style,
                atom_selection=_parse_atom_selection(root["atom_selection"]),
            ),
        )
    except PresetError:
        raise
    except LocalizedError as exc:
        raise PresetError(
            f"预设内容无效：{exc}",
            message_key=exc.message_key,
            message_params=exc.message_params,
        ) from exc
    except (KeyError, TypeError, ValueError) as exc:
        raise PresetError(
            f"预设内容无效：{exc}",
            message_key="preset.invalid_value_type",
            message_params={"error_type": type(exc).__name__},
        ) from exc


def load_default_style() -> StylePreset:
    """通过与用户上传相同的严格解析器读取内置默认风格。"""
    path = Path(__file__).with_name("defaults") / "default_style.meia.json"
    try:
        preset = parse_preset(path.read_bytes())
    except PresetError:
        raise
    except OSError as exc:
        raise PresetError(
            f"默认风格无法读取：{path.resolve()}：{exc}",
            message_key="preset.default_read_failed",
            message_params={"path": str(path.resolve()), "detail": str(exc)},
        ) from exc
    if not isinstance(preset, StylePreset):
        raise PresetError(
            f"默认风格类型错误：{path.resolve()}",
            message_key="preset.default_kind_invalid",
            message_params={"path": str(path.resolve())},
        )
    return preset


def apply_style_preset(
    current_state: VisualizationState,
    preset: StylePreset,
    atoms: Atoms,
) -> VisualizationState:
    if not isinstance(preset, StylePreset):
        raise PresetError(
            "只能应用 StylePreset 到当前结构",
            message_key="preset.type_required",
            message_params={"field": "preset", "expected": "StylePreset"},
        )
    try:
        return apply_portable_style(current_state, preset.style, atoms)
    except LocalizedError as exc:
        raise PresetError(
            f"通用风格无法应用：{exc}",
            message_key=exc.message_key or "preset.apply_failed",
            message_params=exc.message_params,
        ) from exc
    except (TypeError, ValueError) as exc:
        raise PresetError(
            f"通用风格无法应用：{exc}",
            message_key="preset.apply_failed",
        ) from exc


def apply_workspace_snapshot(
    snapshot: WorkspaceSnapshot,
) -> tuple[Atoms, str, VisualizationState]:
    if not isinstance(snapshot, WorkspaceSnapshot):
        raise PresetError(
            "只能应用 WorkspaceSnapshot",
            message_key="preset.type_required",
            message_params={"field": "snapshot", "expected": "WorkspaceSnapshot"},
        )
    return snapshot.structure.to_atoms(), snapshot.structure.source_name, snapshot.state
