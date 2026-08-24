"""原子显示半径与化学键体宽度的双尺寸档案。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
from ase.data import atomic_numbers, covalent_radii

from .i18n import LocalizedError


DISPLAY_RADIUS_EDIT_ABS_TOLERANCE = 1.0e-9


def _positive_number(
    value: object,
    label: str,
    field: str,
    *,
    message_key: str | None = None,
    message_params: Mapping[str, object] | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label}必须是数值")
    number = float(value)
    if not math.isfinite(number):
        raise LocalizedError(
            f"{label}必须是有限数值；收到 {value!r}",
            message_key=message_key or "atom.value_finite",
            message_params=message_params
            or {"field": field, "value": repr(value)},
        )
    if number <= 0:
        raise LocalizedError(
            f"{label}必须大于 0；收到 {value!r}",
            message_key=message_key or "atom.value_positive",
            message_params=message_params
            or {"field": field, "value": repr(value)},
        )
    return number


def _frozen_radius_map(
    values: Mapping[str, float],
    label: str,
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{label}必须是映射")
    normalized: dict[str, float] = {}
    for symbol, value in values.items():
        if symbol not in atomic_numbers or symbol == "X":
            raise LocalizedError(
                f"{label}包含非法元素：{symbol!r}",
                message_key="atom.invalid_element",
                message_params={"symbol": repr(symbol)},
            )
        normalized[symbol] = _positive_number(
            value,
            f"{label}[{symbol}]",
            f"reference_overrides_angstrom[{symbol}]",
        )
    return MappingProxyType(normalized)


class RadiusMode(str, Enum):
    """原子最终显示半径的参考方案。"""

    COVALENT = "covalent"
    UNIFORM = "uniform"


@dataclass(frozen=True)
class CovalentSizeProfile:
    """以 ASE 共价半径为参考的显示尺寸档案。"""

    global_scale: float = 0.6
    reference_overrides_angstrom: Mapping[str, float] = field(default_factory=dict)
    bond_width_ratio: float = 0.45

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_scale",
            _positive_number(
                self.global_scale,
                "共价半径全局倍率",
                "covalent.global_scale",
            ),
        )
        object.__setattr__(
            self,
            "reference_overrides_angstrom",
            _frozen_radius_map(
                self.reference_overrides_angstrom,
                "共价半径参考覆盖",
            ),
        )
        object.__setattr__(
            self,
            "bond_width_ratio",
            _positive_number(
                self.bond_width_ratio,
                "共价档案键宽比例",
                "covalent.bond_width_ratio",
            ),
        )


@dataclass(frozen=True)
class UniformSizeProfile:
    """以统一参考半径为基础的显示尺寸档案。"""

    global_scale: float = 1.0
    reference_radius_angstrom: float = 0.35
    reference_overrides_angstrom: Mapping[str, float] = field(default_factory=dict)
    bond_width_ratio: float = 0.45

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "global_scale",
            _positive_number(
                self.global_scale,
                "相等半径全局倍率",
                "uniform.global_scale",
            ),
        )
        object.__setattr__(
            self,
            "reference_radius_angstrom",
            _positive_number(
                self.reference_radius_angstrom,
                "相等半径统一参考值",
                "uniform.reference_radius_angstrom",
            ),
        )
        object.__setattr__(
            self,
            "reference_overrides_angstrom",
            _frozen_radius_map(
                self.reference_overrides_angstrom,
                "相等半径参考覆盖",
            ),
        )
        object.__setattr__(
            self,
            "bond_width_ratio",
            _positive_number(
                self.bond_width_ratio,
                "相等档案键宽比例",
                "uniform.bond_width_ratio",
            ),
        )


@dataclass(frozen=True)
class SizeProfileSettings:
    """同时持有两套独立尺寸档案和当前生效方案。"""

    active_mode: RadiusMode = RadiusMode.COVALENT
    covalent: CovalentSizeProfile = field(default_factory=CovalentSizeProfile)
    uniform: UniformSizeProfile = field(default_factory=UniformSizeProfile)

    def __post_init__(self) -> None:
        try:
            mode = RadiusMode(self.active_mode)
        except (TypeError, ValueError) as exc:
            raise LocalizedError(
                f"非法原子半径模式：{self.active_mode!r}",
                message_key="atom.invalid_radius_mode",
                message_params={"value": repr(self.active_mode)},
            ) from exc
        if not isinstance(self.covalent, CovalentSizeProfile):
            raise TypeError("共价尺寸档案必须是 CovalentSizeProfile")
        if not isinstance(self.uniform, UniformSizeProfile):
            raise TypeError("相等尺寸档案必须是 UniformSizeProfile")
        object.__setattr__(self, "active_mode", mode)


def _profile_for_mode(
    settings: SizeProfileSettings,
    mode: RadiusMode | str | None = None,
) -> CovalentSizeProfile | UniformSizeProfile:
    if not isinstance(settings, SizeProfileSettings):
        raise TypeError("尺寸档案设置必须是 SizeProfileSettings")
    selected_mode = settings.active_mode if mode is None else RadiusMode(mode)
    return settings.covalent if selected_mode is RadiusMode.COVALENT else settings.uniform


def _default_reference_radius(mode: RadiusMode, profile: object, symbol: str) -> float:
    if symbol not in atomic_numbers or symbol == "X":
        raise LocalizedError(
            f"非法元素符号：{symbol!r}",
            message_key="atom.invalid_element",
            message_params={"symbol": repr(symbol)},
        )
    if mode is RadiusMode.COVALENT:
        return float(covalent_radii[atomic_numbers[symbol]])
    if not isinstance(profile, UniformSizeProfile):
        raise TypeError("相等半径解析必须使用 UniformSizeProfile")
    return profile.reference_radius_angstrom


def resolve_display_radii(
    settings: SizeProfileSettings,
    symbols: Iterable[str],
    *,
    mode: RadiusMode | str | None = None,
) -> np.ndarray:
    """解析指定档案的最终显示半径，单位 Å。"""
    selected_mode = settings.active_mode if mode is None else RadiusMode(mode)
    profile = _profile_for_mode(settings, selected_mode)
    symbol_list = tuple(symbols)
    if not symbol_list:
        return np.empty(0, dtype=float)
    return np.asarray(
        [
            profile.reference_overrides_angstrom.get(
                symbol,
                _default_reference_radius(selected_mode, profile, symbol),
            )
            * profile.global_scale
            for symbol in symbol_list
        ],
        dtype=float,
    )


def resolve_active_bond_width(settings: SizeProfileSettings) -> float:
    """返回当前生效档案的键体宽度比例。"""
    return _profile_for_mode(settings).bond_width_ratio


def apply_size_profile_edits(
    current: SizeProfileSettings,
    *,
    mode: RadiusMode | str,
    global_scale: float,
    uniform_reference_radius_angstrom: float | None,
    submitted_display_radii_angstrom: Mapping[str, float],
) -> SizeProfileSettings:
    """应用目标档案草稿，仅把明确提交的元素值解释为元素覆盖。"""
    if not isinstance(current, SizeProfileSettings):
        raise TypeError("当前尺寸档案必须是 SizeProfileSettings")
    if not isinstance(submitted_display_radii_angstrom, Mapping):
        raise TypeError("提交的显示半径必须是映射")
    selected_mode = RadiusMode(mode)
    scale = _positive_number(global_scale, "原子全局倍率", "global_scale")
    old_profile = _profile_for_mode(current, selected_mode)

    if selected_mode is RadiusMode.COVALENT:
        candidate: CovalentSizeProfile | UniformSizeProfile = replace(
            old_profile,
            global_scale=scale,
        )
    else:
        if uniform_reference_radius_angstrom is None:
            raise ValueError("相等半径档案必须提供统一参考值")
        candidate = replace(
            old_profile,
            global_scale=scale,
            reference_radius_angstrom=uniform_reference_radius_angstrom,
        )

    overrides = dict(candidate.reference_overrides_angstrom)
    for symbol, submitted in submitted_display_radii_angstrom.items():
        if symbol not in atomic_numbers or symbol == "X":
            raise LocalizedError(
                f"提交显示半径包含非法元素：{symbol!r}",
                message_key="atom.invalid_element",
                message_params={"symbol": repr(symbol)},
            )
        submitted_radius = _positive_number(
            submitted,
            f"提交显示半径[{symbol}]",
            f"display radius for {symbol}",
            message_key="atom.invalid_radius",
            message_params={"symbol": symbol, "value": repr(submitted)},
        )
        baseline_radius = (
            _default_reference_radius(selected_mode, candidate, symbol) * scale
        )
        if math.isclose(
            submitted_radius,
            baseline_radius,
            rel_tol=0.0,
            abs_tol=DISPLAY_RADIUS_EDIT_ABS_TOLERANCE,
        ):
            overrides.pop(symbol, None)
        else:
            overrides[symbol] = submitted_radius / scale

    candidate = replace(candidate, reference_overrides_angstrom=overrides)
    return SizeProfileSettings(
        active_mode=selected_mode,
        covalent=(
            candidate
            if selected_mode is RadiusMode.COVALENT
            else current.covalent
        ),
        uniform=(
            candidate
            if selected_mode is RadiusMode.UNIFORM
            else current.uniform
        ),
    )


def replace_active_bond_width(
    settings: SizeProfileSettings,
    bond_width_ratio: float,
) -> SizeProfileSettings:
    """只替换当前生效档案的键体宽度比例。"""
    width = _positive_number(bond_width_ratio, "键宽比例", "bond_width_ratio")
    if settings.active_mode is RadiusMode.COVALENT:
        return replace(
            settings,
            covalent=replace(settings.covalent, bond_width_ratio=width),
        )
    return replace(
        settings,
        uniform=replace(settings.uniform, bond_width_ratio=width),
    )
