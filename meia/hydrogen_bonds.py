"""周期 O–H···O 氢键候选、显示实例与二维投影几何。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list

from .atom_styles import (
    AtomSelectionSettings,
    apply_color_strength,
    normalize_color_strength,
)
from .bond_rules import OverrideVisibility, ResolvedBond
from .bond_segments import clip_bond_to_spheres
from .periodic_display import AtomInstanceKey, LatticeShift, PeriodicDisplay
from .projection import ProjectionResult
from .i18n import LocalizedError


HYDROGEN_BOND_MAX_DISTANCE = 2.5
HYDROGEN_BOND_MIN_ANGLE = 120.0
HYDROGEN_BOND_COLOR = "#78909C"
HYDROGEN_BOND_2D_WIDTH = 1.0
HYDROGEN_BOND_3D_WIDTH = 3.0


@dataclass(frozen=True)
class HydrogenBondSettings:
    """氢键显示与几何筛选的已应用全局设置。"""

    draw: bool = True
    max_hydrogen_oxygen_distance: float = HYDROGEN_BOND_MAX_DISTANCE
    min_angle_degrees: float = HYDROGEN_BOND_MIN_ANGLE

    def __post_init__(self) -> None:
        if not isinstance(self.draw, bool):
            raise LocalizedError(
                "氢键显示开关必须是布尔值",
                message_key="hydrogen.draw_boolean",
                message_params={"value": repr(self.draw)},
            )
        maximum = _finite_threshold(
            self.max_hydrogen_oxygen_distance,
            "H···O 最大距离",
            "maximum H···O distance",
        )
        angle = _finite_threshold(
            self.min_angle_degrees,
            "O–H···O 最小夹角",
            "minimum O–H···O angle",
        )
        if maximum <= 0:
            raise LocalizedError(
                "H···O 最大距离必须大于 0 Å",
                message_key="hydrogen.distance_positive",
                message_params={"value": maximum},
            )
        if not 0 <= angle <= 180:
            raise LocalizedError(
                "O–H···O 最小夹角必须在 0°–180° 之间",
                message_key="hydrogen.angle_range",
                message_params={"value": angle},
            )
        object.__setattr__(self, "max_hydrogen_oxygen_distance", maximum)
        object.__setattr__(self, "min_angle_degrees", angle)


@dataclass(frozen=True)
class HydrogenBondCandidate:
    """一根满足几何硬条件、保留周期像身份的 O–H···O 候选。"""

    donor_oxygen: int
    hydrogen: int
    acceptor_oxygen: int
    donor_oxygen_offset_from_hydrogen: LatticeShift
    acceptor_offset_from_hydrogen: LatticeShift
    hydrogen_acceptor_distance: float
    angle_degrees: float
    hydrogen_bond_id: str


# v0.7 公共名称继续表示源几何候选；3D 周期实例将在 Task 5 消费新接口。
HydrogenBond = HydrogenBondCandidate


@dataclass(frozen=True)
class DisplayHydrogenBond:
    """一个请求范围内、由三个原子显示 key 确定的氢键实例。"""

    candidate: HydrogenBondCandidate
    donor_oxygen_key: AtomInstanceKey
    hydrogen_key: AtomInstanceKey
    acceptor_oxygen_key: AtomInstanceKey
    instance_id: str
    color: str
    color_strength: float
    visible: bool
    visibility_source: str


@dataclass(frozen=True)
class HydrogenBondGeometry:
    """一根周期氢键在 2D 输出中的可见线段与显示像元数据。"""

    start: np.ndarray
    end: np.ndarray
    group_depth: float
    hydrogen_bond: DisplayHydrogenBond
    color: str
    donor_oxygen_image_shift: LatticeShift
    hydrogen_image_shift: LatticeShift
    acceptor_oxygen_image_shift: LatticeShift


def _finite_threshold(value: float, label: str, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise LocalizedError(
            f"{label}必须是数值",
            message_key="hydrogen.value_numeric",
            message_params={"field": field, "value": repr(value)},
        )
    result = float(value)
    if not math.isfinite(result):
        raise LocalizedError(
            f"{label}必须是有限数值",
            message_key="hydrogen.value_finite",
            message_params={"field": field, "value": repr(value)},
        )
    return result


def _angle_degrees(vector_a: np.ndarray, vector_b: np.ndarray) -> float | None:
    norm_a = float(np.linalg.norm(vector_a))
    norm_b = float(np.linalg.norm(vector_b))
    if norm_a <= np.finfo(float).eps or norm_b <= np.finfo(float).eps:
        return None
    cosine = float(np.dot(vector_a, vector_b) / (norm_a * norm_b))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _shift(values: Sequence[int]) -> LatticeShift:
    return tuple(int(value) for value in values)  # type: ignore[return-value]


def _add_shifts(*shifts: LatticeShift) -> LatticeShift:
    return tuple(sum(values) for values in zip(*shifts))  # type: ignore[return-value]


def _negate_shift(shift: LatticeShift) -> LatticeShift:
    return tuple(-value for value in shift)  # type: ignore[return-value]


def _shift_token(shift: LatticeShift) -> str:
    return "_".join(str(value) for value in shift)


def resolve_hydrogen_bond_candidates(
    atoms: Atoms,
    matched_covalent_bonds: Sequence[ResolvedBond],
    *,
    max_hydrogen_oxygen_distance: float = HYDROGEN_BOND_MAX_DISTANCE,
    min_angle_degrees: float = HYDROGEN_BOND_MIN_ANGLE,
) -> tuple[HydrogenBondCandidate, ...]:
    """按 matched O–H、H···O 距离与角度解析周期氢键候选。

    共价键的 ``visible`` 不参与供体识别；候选也不使用 O···O cutoff。
    所有偏移均表示相应 O 的周期像相对于源 H 坐标的整数晶格平移。
    """
    if not isinstance(atoms, Atoms):
        raise TypeError("氢键候选必须绑定 ASE Atoms")
    maximum = _finite_threshold(
        max_hydrogen_oxygen_distance,
        "H···O 最大距离",
        "maximum H···O distance",
    )
    minimum_angle = _finite_threshold(
        min_angle_degrees,
        "O–H···O 最小夹角",
        "minimum O–H···O angle",
    )
    if maximum <= 0:
        raise LocalizedError(
            "H···O 最大距离必须大于 0 Å",
            message_key="hydrogen.distance_positive",
            message_params={"value": maximum},
        )
    if not 0 <= minimum_angle <= 180:
        raise LocalizedError(
            "O–H···O 最小夹角必须在 0°–180° 之间",
            message_key="hydrogen.angle_range",
            message_params={"value": minimum_angle},
        )

    bonds = tuple(matched_covalent_bonds)
    if not all(isinstance(bond, ResolvedBond) for bond in bonds):
        raise TypeError("氢键供体必须来自 ResolvedBond")
    if len(atoms) == 0 or not bonds:
        return ()

    symbols = atoms.get_chemical_symbols()
    positions = np.asarray(atoms.positions, dtype=float)
    cell = np.asarray(atoms.cell, dtype=float)
    donors: set[tuple[int, int, LatticeShift]] = set()
    for bond in bonds:
        endpoints = ((bond.i, symbols[bond.i]), (bond.j, symbols[bond.j]))
        hydrogens = [index for index, symbol in endpoints if symbol == "H"]
        oxygens = [index for index, symbol in endpoints if symbol == "O"]
        if len(hydrogens) != 1 or len(oxygens) != 1:
            continue
        hydrogen = hydrogens[0]
        donor_oxygen = oxygens[0]
        bond_offset = _shift(bond.offset)
        donor_offset = (
            bond_offset if bond.i == hydrogen else _negate_shift(bond_offset)
        )
        donors.add((donor_oxygen, hydrogen, donor_offset))

    if not donors:
        return ()

    cutoff = float(np.nextafter(maximum, math.inf))
    first, second, distances, offsets = neighbor_list(
        "ijdS",
        atoms,
        cutoff=cutoff,
        self_interaction=False,
    )
    acceptors_by_hydrogen: dict[
        int,
        list[tuple[int, LatticeShift, float]],
    ] = {}
    for raw_i, raw_j, raw_distance, raw_offset in zip(
        first,
        second,
        distances,
        offsets,
    ):
        hydrogen = int(raw_i)
        acceptor = int(raw_j)
        if symbols[hydrogen] != "H" or symbols[acceptor] != "O":
            continue
        distance = float(raw_distance)
        if distance > maximum:
            continue
        acceptors_by_hydrogen.setdefault(hydrogen, []).append(
            (acceptor, _shift(raw_offset), distance)
        )

    pending: dict[
        tuple[int, int, int, LatticeShift, LatticeShift],
        tuple[float, float],
    ] = {}
    for donor_oxygen, hydrogen, donor_offset in sorted(donors):
        donor_vector = (
            positions[donor_oxygen]
            + np.dot(np.asarray(donor_offset, dtype=int), cell)
            - positions[hydrogen]
        )
        for acceptor_oxygen, acceptor_offset, distance in acceptors_by_hydrogen.get(
            hydrogen,
            (),
        ):
            if (
                acceptor_oxygen == donor_oxygen
                and acceptor_offset == donor_offset
            ):
                continue
            acceptor_vector = (
                positions[acceptor_oxygen]
                + np.dot(np.asarray(acceptor_offset, dtype=int), cell)
                - positions[hydrogen]
            )
            angle = _angle_degrees(donor_vector, acceptor_vector)
            # 允许由三角函数舍入造成的一个 ULP 临界误差，物理门槛仍是 120°。
            if angle is None or angle < np.nextafter(minimum_angle, -math.inf):
                continue
            key = (
                donor_oxygen,
                hydrogen,
                acceptor_oxygen,
                donor_offset,
                acceptor_offset,
            )
            previous = pending.get(key)
            if previous is None or distance < previous[0]:
                pending[key] = (distance, angle)

    candidates = []
    for key in sorted(pending):
        (
            donor_oxygen,
            hydrogen,
            acceptor_oxygen,
            donor_offset,
            acceptor_offset,
        ) = key
        distance, angle = pending[key]
        candidates.append(
            HydrogenBondCandidate(
                donor_oxygen=donor_oxygen,
                hydrogen=hydrogen,
                acceptor_oxygen=acceptor_oxygen,
                donor_oxygen_offset_from_hydrogen=donor_offset,
                acceptor_offset_from_hydrogen=acceptor_offset,
                hydrogen_acceptor_distance=distance,
                angle_degrees=angle,
                hydrogen_bond_id=(
                    f"hydrogen_bond_O{donor_oxygen + 1}_H{hydrogen + 1}_"
                    f"O{acceptor_oxygen + 1}"
                    f"__donor_offset_{_shift_token(donor_offset)}"
                    f"__acceptor_offset_{_shift_token(acceptor_offset)}"
                ),
            )
        )
    return tuple(candidates)


def resolve_hydrogen_bonds(
    atoms: Atoms,
    covalent_bonds: Sequence[ResolvedBond],
    *,
    max_hydrogen_oxygen_distance: float = HYDROGEN_BOND_MAX_DISTANCE,
    min_angle_degrees: float = HYDROGEN_BOND_MIN_ANGLE,
) -> tuple[HydrogenBondCandidate, ...]:
    """兼容旧入口；返回保留周期偏移的源候选。"""
    return resolve_hydrogen_bond_candidates(
        atoms,
        covalent_bonds,
        max_hydrogen_oxygen_distance=max_hydrogen_oxygen_distance,
        min_angle_degrees=min_angle_degrees,
    )


def instantiate_periodic_hydrogen_bonds(
    atoms: Atoms,
    periodic_display: PeriodicDisplay,
    candidates: Sequence[HydrogenBondCandidate],
    atom_selection: AtomSelectionSettings,
    color_strengths: Mapping[int, float],
) -> tuple[DisplayHydrogenBond, ...]:
    """把几何候选映射到已请求的三个周期原子显示实例。"""
    if not isinstance(atoms, Atoms):
        raise TypeError("周期氢键实例必须绑定 ASE Atoms")
    if not isinstance(periodic_display, PeriodicDisplay):
        raise TypeError("周期氢键实例必须使用 PeriodicDisplay")
    if not isinstance(atom_selection, AtomSelectionSettings):
        raise TypeError("周期氢键实例必须使用 AtomSelectionSettings")

    hidden = {item.atom_index for item in atom_selection.hidden_atoms}
    overrides = {
        item.atom_index: item.visibility
        for item in atom_selection.hydrogen_bond_overrides
    }
    strengths = {
        int(index): normalize_color_strength(value)
        for index, value in color_strengths.items()
    }
    base_shifts = periodic_display.base_image_shifts
    seen: set[
        tuple[str, AtomInstanceKey, AtomInstanceKey, AtomInstanceKey]
    ] = set()
    instances = []
    for candidate in candidates:
        participants = (
            candidate.donor_oxygen,
            candidate.hydrogen,
            candidate.acceptor_oxygen,
        )
        if any(index in hidden for index in participants):
            continue
        participant_states = tuple(
            overrides.get(index, OverrideVisibility.INHERIT)
            for index in participants
        )
        if OverrideVisibility.HIDE in participant_states:
            visible = False
            visibility_source = "atom_hide"
        elif OverrideVisibility.SHOW in participant_states:
            visible = True
            visibility_source = "atom_show"
        else:
            visible = True
            visibility_source = "default_visible"

        for hydrogen_replica in periodic_display.replica_translations:
            donor_replica = _add_shifts(
                hydrogen_replica,
                base_shifts[candidate.hydrogen],
                candidate.donor_oxygen_offset_from_hydrogen,
                _negate_shift(base_shifts[candidate.donor_oxygen]),
            )
            acceptor_replica = _add_shifts(
                hydrogen_replica,
                base_shifts[candidate.hydrogen],
                candidate.acceptor_offset_from_hydrogen,
                _negate_shift(base_shifts[candidate.acceptor_oxygen]),
            )
            donor_key = (candidate.donor_oxygen, donor_replica)
            hydrogen_key = (candidate.hydrogen, hydrogen_replica)
            acceptor_key = (candidate.acceptor_oxygen, acceptor_replica)
            if any(
                key not in periodic_display.atom_by_key
                for key in (donor_key, hydrogen_key, acceptor_key)
            ):
                continue
            identity = (
                candidate.hydrogen_bond_id,
                donor_key,
                hydrogen_key,
                acceptor_key,
            )
            if identity in seen:
                continue
            seen.add(identity)
            strength = min(strengths.get(index, 1.0) for index in participants)
            donor_instance = periodic_display.atom_by_key[donor_key]
            hydrogen_instance = periodic_display.atom_by_key[hydrogen_key]
            acceptor_instance = periodic_display.atom_by_key[acceptor_key]
            instances.append(
                DisplayHydrogenBond(
                    candidate=candidate,
                    donor_oxygen_key=donor_key,
                    hydrogen_key=hydrogen_key,
                    acceptor_oxygen_key=acceptor_key,
                    instance_id=(
                        f"{candidate.hydrogen_bond_id}"
                        f"__donor_image_{_shift_token(donor_instance.image_shift)}"
                        f"__hydrogen_image_{_shift_token(hydrogen_instance.image_shift)}"
                        f"__acceptor_image_{_shift_token(acceptor_instance.image_shift)}"
                    ),
                    color=apply_color_strength(HYDROGEN_BOND_COLOR, strength),
                    color_strength=strength,
                    visible=visible,
                    visibility_source=visibility_source,
                )
            )
    return tuple(instances)


def compute_hydrogen_bond_geometries(
    hydrogen_bonds: Sequence[DisplayHydrogenBond],
    projection: ProjectionResult,
) -> tuple[HydrogenBondGeometry, ...]:
    """把显示 H···O 中心线裁剪到两个周期原子实例的 2D 球面。"""
    geometries = []
    for hydrogen_bond in hydrogen_bonds:
        if not hydrogen_bond.visible:
            continue
        donor_row = projection.instance_index_by_key.get(
            hydrogen_bond.donor_oxygen_key
        )
        hydrogen_row = projection.instance_index_by_key.get(
            hydrogen_bond.hydrogen_key
        )
        acceptor_row = projection.instance_index_by_key.get(
            hydrogen_bond.acceptor_oxygen_key
        )
        if donor_row is None or hydrogen_row is None or acceptor_row is None:
            continue
        segment = clip_bond_to_spheres(
            projection.positions_2d[hydrogen_row],
            projection.positions_2d[acceptor_row],
            projection.radii_2d[hydrogen_row],
            projection.radii_2d[acceptor_row],
        )
        if segment is None:
            continue
        geometries.append(
            HydrogenBondGeometry(
                start=segment.start,
                end=segment.end,
                group_depth=float(
                    (
                        projection.depths[hydrogen_row]
                        + projection.depths[acceptor_row]
                    )
                    / 2
                ),
                hydrogen_bond=hydrogen_bond,
                color=hydrogen_bond.color,
                donor_oxygen_image_shift=_shift(
                    projection.image_shifts[donor_row]
                ),
                hydrogen_image_shift=_shift(projection.image_shifts[hydrogen_row]),
                acceptor_oxygen_image_shift=_shift(
                    projection.image_shifts[acceptor_row]
                ),
            )
        )
    return tuple(geometries)
