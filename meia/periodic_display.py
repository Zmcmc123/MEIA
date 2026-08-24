"""周期显示状态与元素无关的成键图展开。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from itertools import product
from numbers import Integral
from types import MappingProxyType
from typing import Iterable, Mapping

import numpy as np
from ase import Atoms

from .bond_rules import ElementPair, ResolvedBond
from .i18n import LocalizedError


LatticeShift = tuple[int, int, int]
AtomInstanceKey = tuple[int, LatticeShift]
MAX_PERIODIC_ATOM_INSTANCES = 50_000
_ZERO_SHIFT: LatticeShift = (0, 0, 0)


@dataclass(frozen=True)
class PeriodicRange:
    """一个晶格轴上左闭右开的周期平移范围。"""

    start: int = 0
    end: int = 1

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.start, self.end)
        ):
            raise LocalizedError(
                "周期边界必须是整数",
                message_key="periodic.boundary_integer",
                message_params={
                    "start": repr(self.start),
                    "end": repr(self.end),
                },
            )
        if self.end <= self.start:
            raise LocalizedError(
                "周期边界终点必须大于起点",
                message_key="periodic.range_invalid",
                message_params={"start": self.start, "end": self.end},
            )


@dataclass(frozen=True)
class CellPeriodicSettings:
    """晶胞图层与三个晶格轴的周期显示状态。"""

    show_unit_cell: int = 2
    unwrap_bonded_groups: bool = True
    a: PeriodicRange = field(default_factory=PeriodicRange)
    b: PeriodicRange = field(default_factory=PeriodicRange)
    c: PeriodicRange = field(default_factory=PeriodicRange)

    def __post_init__(self) -> None:
        if (
            isinstance(self.show_unit_cell, bool)
            or not isinstance(self.show_unit_cell, int)
            or self.show_unit_cell not in {0, 1, 2}
        ):
            raise LocalizedError(
                "晶胞显示模式只能是 0、1 或 2",
                message_key="periodic.invalid_cell_mode",
                message_params={"value": repr(self.show_unit_cell)},
            )
        if not isinstance(self.unwrap_bonded_groups, bool):
            raise LocalizedError(
                "跨边界成键组展开开关必须是布尔值",
                message_key="periodic.unwrap_boolean",
                message_params={"value": repr(self.unwrap_bonded_groups)},
            )
        if not all(
            isinstance(value, PeriodicRange) for value in (self.a, self.b, self.c)
        ):
            raise TypeError("周期范围必须是 PeriodicRange")


@dataclass(frozen=True)
class AtomDisplayInstance:
    """一个请求晶胞副本中的原子显示实例。"""

    source_atom_index: int
    replica_translation: LatticeShift
    image_shift: LatticeShift
    position: np.ndarray
    instance_id: str


@dataclass(frozen=True)
class BondDisplayInstance:
    """两个已请求原子镜像之间的成键实例。"""

    source_bond: ResolvedBond
    atom_i_key: AtomInstanceKey
    atom_j_key: AtomInstanceKey
    bond_instance_id: str


@dataclass(frozen=True)
class PeriodicDisplayDiagnostic:
    """周期图无法唯一展开时的可追溯诊断。"""

    code: str
    atom_indices: tuple[int, ...]
    conflicting_bond_ids: tuple[str, ...]
    conflicting_element_pairs: tuple[ElementPair, ...] = ()

    @property
    def bond_ids(self) -> tuple[str, ...]:
        """为调用端提供简短的键标识别名。"""
        return self.conflicting_bond_ids


@dataclass(frozen=True)
class PeriodicDisplay:
    """周期显示引擎的完整、确定性输出。"""

    base_image_shifts: tuple[LatticeShift, ...]
    base_shift_reasons: tuple[str, ...]
    replica_translations: tuple[LatticeShift, ...]
    atom_instances: tuple[AtomDisplayInstance, ...]
    bond_instances: tuple[BondDisplayInstance, ...]
    diagnostics: tuple[PeriodicDisplayDiagnostic, ...]
    atom_by_key: Mapping[AtomInstanceKey, AtomDisplayInstance]


@dataclass(frozen=True)
class _GraphEdge:
    edge_id: int
    bond: ResolvedBond
    i: int
    j: int
    offset: LatticeShift


_AdjacencyEntry = tuple[int, int, LatticeShift]


def estimate_periodic_atom_instances(
    atoms: Atoms,
    settings: CellPeriodicSettings,
) -> int:
    """估算三个周期范围内需要显示的原子实例总数。"""
    if not isinstance(atoms, Atoms):
        raise TypeError("周期显示必须绑定 ASE Atoms")
    if not isinstance(settings, CellPeriodicSettings):
        raise TypeError("周期显示设置必须是 CellPeriodicSettings")
    copies = 1
    for axis_range in (settings.a, settings.b, settings.c):
        copies *= axis_range.end - axis_range.start
    return len(atoms) * copies


def normalize_periodic_settings(
    atoms: Atoms,
    settings: CellPeriodicSettings,
) -> CellPeriodicSettings:
    """禁用非 PBC 轴的重复，并拒绝超过安全上限的实例数量。"""
    if not isinstance(atoms, Atoms):
        raise TypeError("周期显示必须绑定 ASE Atoms")
    if not isinstance(settings, CellPeriodicSettings):
        raise TypeError("周期显示设置必须是 CellPeriodicSettings")
    normalized = replace(
        settings,
        **{
            axis: value if bool(atoms.pbc[index]) else PeriodicRange()
            for index, (axis, value) in enumerate(
                (("a", settings.a), ("b", settings.b), ("c", settings.c))
            )
        },
    )
    if estimate_periodic_atom_instances(atoms, normalized) > MAX_PERIODIC_ATOM_INSTANCES:
        estimated = estimate_periodic_atom_instances(atoms, normalized)
        raise LocalizedError(
            "预计显示原子数超过 50,000",
            message_key="periodic.atom_limit",
            message_params={
                "count": f"{estimated:,}",
                "limit": f"{MAX_PERIODIC_ATOM_INSTANCES:,}",
            },
        )
    return normalized


def _add_shifts(*shifts: LatticeShift) -> LatticeShift:
    return tuple(sum(values) for values in zip(*shifts))  # type: ignore[return-value]


def _negate_shift(shift: LatticeShift) -> LatticeShift:
    return tuple(-value for value in shift)  # type: ignore[return-value]


def _subtract_shifts(first: LatticeShift, second: LatticeShift) -> LatticeShift:
    return _add_shifts(first, _negate_shift(second))


def _validate_graph_edges(
    atom_count: int,
    matched_bonds: Iterable[ResolvedBond],
) -> tuple[_GraphEdge, ...]:
    try:
        bonds = tuple(matched_bonds)
    except TypeError as exc:
        raise TypeError("已匹配化学键必须可迭代") from exc

    validated: list[tuple[ResolvedBond, int, int, LatticeShift]] = []
    for bond in bonds:
        if not isinstance(bond, ResolvedBond):
            raise TypeError("周期图边必须是 ResolvedBond")
        if any(
            isinstance(index, bool) or not isinstance(index, Integral)
            for index in (bond.i, bond.j)
        ):
            raise TypeError("化学键原子索引必须是整数")
        i, j = int(bond.i), int(bond.j)
        if not (0 <= i < atom_count and 0 <= j < atom_count):
            raise ValueError("化学键原子索引超出当前构型范围")
        try:
            raw_offset = tuple(bond.offset)
        except TypeError as exc:
            raise TypeError("化学键周期偏移必须可迭代") from exc
        if len(raw_offset) != 3:
            raise ValueError("化学键周期偏移必须恰好包含三个整数")
        if any(
            isinstance(value, bool) or not isinstance(value, Integral)
            for value in raw_offset
        ):
            raise TypeError("化学键周期偏移必须是三个整数")
        offset = tuple(int(value) for value in raw_offset)
        validated.append((bond, i, j, offset))  # type: ignore[arg-type]

    validated.sort(
        key=lambda item: (
            item[1],
            item[2],
            item[3],
            item[0].bond_id,
        )
    )
    return tuple(
        _GraphEdge(edge_id, bond, i, j, offset)
        for edge_id, (bond, i, j, offset) in enumerate(validated)
    )


def _build_adjacency(
    atom_count: int,
    edges: tuple[_GraphEdge, ...],
) -> tuple[tuple[_AdjacencyEntry, ...], ...]:
    adjacency: list[list[_AdjacencyEntry]] = [[] for _ in range(atom_count)]
    for edge in edges:
        adjacency[edge.i].append((edge.edge_id, edge.j, edge.offset))
        adjacency[edge.j].append(
            (edge.edge_id, edge.i, _negate_shift(edge.offset))
        )
    return tuple(
        tuple(sorted(entries, key=lambda item: (item[1], item[2], item[0])))
        for entries in adjacency
    )


def _connected_components(
    adjacency: tuple[tuple[_AdjacencyEntry, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    remaining = set(range(len(adjacency)))
    components = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        component = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for _, neighbor, _ in adjacency[current]:
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _eccentricity(
    source: int,
    component_atoms: set[int],
    adjacency: tuple[tuple[_AdjacencyEntry, ...], ...],
) -> int:
    distances = {source: 0}
    queue = deque([source])
    while queue:
        current = queue.popleft()
        for _, neighbor, _ in adjacency[current]:
            if neighbor in component_atoms and neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return max(distances.values(), default=0)


def _graph_center(
    component: tuple[int, ...],
    atomic_numbers: np.ndarray,
    adjacency: tuple[tuple[_AdjacencyEntry, ...], ...],
) -> int:
    component_atoms = set(component)
    return min(
        component,
        key=lambda source: (
            _eccentricity(source, component_atoms, adjacency),
            -int(atomic_numbers[source]),
            source,
        ),
    )


def _propagate_shifts(
    atoms_to_visit: set[int],
    root: int,
    root_shift: LatticeShift,
    adjacency: tuple[tuple[_AdjacencyEntry, ...], ...],
    ignored_edges: set[int] | frozenset[int] = frozenset(),
) -> tuple[dict[int, LatticeShift], bool]:
    shifts = {root: root_shift}
    contradiction = False
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for edge_id, neighbor, signed_offset in adjacency[current]:
            if edge_id in ignored_edges or neighbor not in atoms_to_visit:
                continue
            expected = _add_shifts(shifts[current], signed_offset)
            if neighbor not in shifts:
                shifts[neighbor] = expected
                queue.append(neighbor)
            elif shifts[neighbor] != expected:
                contradiction = True
    return shifts, contradiction


def _find_bridges(
    component: tuple[int, ...],
    adjacency: tuple[tuple[_AdjacencyEntry, ...], ...],
) -> frozenset[int]:
    discovery: dict[int, int] = {}
    low: dict[int, int] = {}
    bridges: set[int] = set()
    root = component[0]
    discovery[root] = 0
    low[root] = 0
    parent_edge: dict[int, int | None] = {root: None}
    parent_atom: dict[int, int] = {}
    timer = 1
    stack: list[tuple[int, int]] = [(root, 0)]

    # The parent relation is an edge id, not an atom id. This is the invariant
    # that lets a parallel edge to the parent lower low[current] normally.
    while stack:
        current, next_entry = stack[-1]
        if next_entry < len(adjacency[current]):
            edge_id, neighbor, _ = adjacency[current][next_entry]
            stack[-1] = (current, next_entry + 1)
            if edge_id == parent_edge[current]:
                continue
            if neighbor in discovery:
                low[current] = min(low[current], discovery[neighbor])
                continue
            discovery[neighbor] = timer
            low[neighbor] = timer
            timer += 1
            parent_edge[neighbor] = edge_id
            parent_atom[neighbor] = current
            stack.append((neighbor, 0))
            continue

        stack.pop()
        incoming_edge = parent_edge[current]
        if incoming_edge is None:
            continue
        parent = parent_atom[current]
        low[parent] = min(low[parent], low[current])
        if low[current] > discovery[parent]:
            bridges.add(incoming_edge)

    return frozenset(bridges)


def _two_edge_connected_blocks(
    component: tuple[int, ...],
    adjacency: tuple[tuple[_AdjacencyEntry, ...], ...],
    bridges: frozenset[int],
) -> tuple[tuple[int, ...], ...]:
    remaining = set(component)
    blocks = []
    while remaining:
        start = min(remaining)
        queue = deque([start])
        remaining.remove(start)
        block = []
        while queue:
            current = queue.popleft()
            block.append(current)
            for edge_id, neighbor, _ in adjacency[current]:
                if edge_id not in bridges and neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
        blocks.append(tuple(sorted(block)))
    return tuple(blocks)


def _block_cycle_residuals(
    block: tuple[int, ...],
    block_edges: tuple[_GraphEdge, ...],
    adjacency: tuple[tuple[_AdjacencyEntry, ...], ...],
) -> tuple[
    dict[int, LatticeShift],
    tuple[tuple[_GraphEdge, LatticeShift], ...],
]:
    block_atoms = set(block)
    block_edge_ids = {edge.edge_id for edge in block_edges}
    root = min(block)
    shifts = {root: _ZERO_SHIFT}
    tree_edges: set[int] = set()
    queue = deque([root])
    while queue:
        current = queue.popleft()
        for edge_id, neighbor, signed_offset in adjacency[current]:
            if (
                edge_id not in block_edge_ids
                or neighbor not in block_atoms
                or neighbor in shifts
            ):
                continue
            shifts[neighbor] = _add_shifts(shifts[current], signed_offset)
            tree_edges.add(edge_id)
            queue.append(neighbor)

    residuals = []
    for edge in block_edges:
        if edge.edge_id in tree_edges:
            continue
        residual = _subtract_shifts(
            _add_shifts(shifts[edge.i], edge.offset),
            shifts[edge.j],
        )
        residuals.append((edge, residual))
    return shifts, tuple(residuals)


def _integer_vector_rank(vectors: Iterable[LatticeShift]) -> int:
    """在有理数域上精确计算三维整数向量张成空间的秩。"""
    nonzero = [vector for vector in vectors if vector != _ZERO_SHIFT]
    if not nonzero:
        return 0
    first = nonzero[0]

    def cross(
        left: LatticeShift,
        right: LatticeShift,
    ) -> LatticeShift:
        return (
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        )

    second = next(
        (vector for vector in nonzero[1:] if cross(first, vector) != _ZERO_SHIFT),
        None,
    )
    if second is None:
        return 1
    normal = cross(first, second)
    if any(sum(a * b for a, b in zip(normal, vector)) != 0 for vector in nonzero):
        return 3
    return 2


def _minimal_periodic_backbone(
    block_count: int,
    periodic_blocks: set[int],
    block_tree: dict[int, list[tuple[int, int, int, int, LatticeShift]]],
) -> set[int]:
    active = set(range(block_count))
    degrees = {block: len(block_tree[block]) for block in active}
    queue = deque(
        sorted(
            block
            for block in active
            if block not in periodic_blocks and degrees[block] <= 1
        )
    )
    while queue:
        block = queue.popleft()
        if block not in active or block in periodic_blocks or degrees[block] > 1:
            continue
        active.remove(block)
        for neighbor, _, _, _, _ in block_tree[block]:
            if neighbor not in active:
                continue
            degrees[neighbor] -= 1
            if neighbor not in periodic_blocks and degrees[neighbor] <= 1:
                queue.append(neighbor)
    return active


def _periodic_component_shifts(
    component: tuple[int, ...],
    edges: tuple[_GraphEdge, ...],
    adjacency: tuple[tuple[_AdjacencyEntry, ...], ...],
) -> tuple[
    dict[int, LatticeShift],
    dict[int, str],
    tuple[PeriodicDisplayDiagnostic, ...],
] | None:
    bridges = _find_bridges(component, adjacency)
    blocks = _two_edge_connected_blocks(component, adjacency, bridges)
    atom_to_block = {
        atom_index: block_index
        for block_index, block in enumerate(blocks)
        for atom_index in block
    }
    edges_by_block: dict[int, list[_GraphEdge]] = {
        block_index: [] for block_index in range(len(blocks))
    }
    for edge in edges:
        if edge.i not in atom_to_block:
            continue
        if edge.edge_id in bridges:
            continue
        block_index = atom_to_block[edge.i]
        # After removing all bridges, every remaining edge must stay inside one
        # block; consequently all winding contradictions live inside blocks.
        if block_index != atom_to_block[edge.j]:
            raise RuntimeError("非桥边跨越二边连通块")
        edges_by_block[block_index].append(edge)

    block_relative_shifts: dict[int, dict[int, LatticeShift]] = {}
    block_is_ambiguous: dict[int, bool] = {}
    periodic_blocks = set()
    diagnostics = []
    for block_index, block in enumerate(blocks):
        block_edges = tuple(edges_by_block[block_index])
        relative_shifts, cycle_residuals = _block_cycle_residuals(
            block,
            block_edges,
            adjacency,
        )
        block_relative_shifts[block_index] = relative_shifts
        cycle_dimension = len(block_edges) - len(block) + 1
        if len(cycle_residuals) != cycle_dimension:
            raise RuntimeError("二边连通块的循环空间维数与非树边不一致")
        residual_span_rank = _integer_vector_rank(
            residual for _, residual in cycle_residuals
        )
        if residual_span_rank > 0:
            periodic_blocks.add(block_index)
        ambiguous = (
            residual_span_rank > 0 and cycle_dimension > residual_span_rank
        )
        block_is_ambiguous[block_index] = ambiguous
        if ambiguous:
            diagnostics.append(
                PeriodicDisplayDiagnostic(
                    "ambiguous_periodic_attachment",
                    block,
                    tuple(edge.bond.bond_id for edge, _ in cycle_residuals),
                    tuple(sorted({edge.bond.pair for edge in block_edges})),
                )
            )
    if not periodic_blocks:
        return None

    block_tree: dict[
        int, list[tuple[int, int, int, int, LatticeShift]]
    ] = {block_index: [] for block_index in range(len(blocks))}
    edge_by_id = {edge.edge_id: edge for edge in edges}
    for edge_id in sorted(bridges):
        edge = edge_by_id[edge_id]
        first_block = atom_to_block[edge.i]
        second_block = atom_to_block[edge.j]
        block_tree[first_block].append(
            (second_block, edge_id, edge.i, edge.j, edge.offset)
        )
        block_tree[second_block].append(
            (first_block, edge_id, edge.j, edge.i, _negate_shift(edge.offset))
        )
    for entries in block_tree.values():
        entries.sort(key=lambda item: (item[0], item[2], item[3], item[1]))

    backbone = _minimal_periodic_backbone(
        len(blocks),
        periodic_blocks,
        block_tree,
    )
    shifts: dict[int, LatticeShift] = {}
    reasons: dict[int, str] = {}
    ambiguous_blocks = {
        block_index
        for block_index, ambiguous in block_is_ambiguous.items()
        if ambiguous
    }
    for block_index in sorted(backbone):
        reason = (
            "ambiguous_periodic_attachment"
            if block_index in ambiguous_blocks
            else "periodic_backbone"
        )
        for atom_index in blocks[block_index]:
            shifts[atom_index] = _ZERO_SHIFT
            reasons[atom_index] = reason

    visited_blocks = set(backbone)
    queue = deque(sorted(backbone))
    while queue:
        source_block = queue.popleft()
        for target_block, _, source_atom, target_atom, signed_offset in block_tree[
            source_block
        ]:
            if target_block in visited_blocks:
                continue
            target_root_shift = _add_shifts(shifts[source_atom], signed_offset)
            relative = block_relative_shifts[target_block]
            adjustment = _subtract_shifts(
                target_root_shift,
                relative[target_atom],
            )
            for atom_index in blocks[target_block]:
                shifts[atom_index] = _add_shifts(relative[atom_index], adjustment)
                reasons[atom_index] = "bridge_unwrap"
            visited_blocks.add(target_block)
            queue.append(target_block)

    return shifts, reasons, tuple(diagnostics)


def _base_image_shifts(
    atoms: Atoms,
    edges: tuple[_GraphEdge, ...],
    adjacency: tuple[tuple[_AdjacencyEntry, ...], ...],
    unwrap_bonded_groups: bool,
) -> tuple[
    tuple[LatticeShift, ...],
    tuple[str, ...],
    tuple[PeriodicDisplayDiagnostic, ...],
]:
    if not unwrap_bonded_groups:
        return (
            (_ZERO_SHIFT,) * len(atoms),
            ("unwrapping_disabled",) * len(atoms),
            (),
        )

    shifts: dict[int, LatticeShift] = {}
    reasons: dict[int, str] = {}
    diagnostics = []
    atomic_numbers = np.asarray(atoms.numbers)
    for component in _connected_components(adjacency):
        periodic = _periodic_component_shifts(component, edges, adjacency)
        if periodic is None:
            root = _graph_center(component, atomic_numbers, adjacency)
            component_shifts, contradiction = _propagate_shifts(
                set(component),
                root,
                _ZERO_SHIFT,
                adjacency,
            )
            if contradiction:  # 二边连通块残差应已覆盖所有矛盾。
                raise RuntimeError("周期图内部矛盾分类失败")
            shifts.update(component_shifts)
            reasons.update({atom_index: "finite_component" for atom_index in component})
            continue
        component_shifts, component_reasons, component_diagnostics = periodic
        shifts.update(component_shifts)
        reasons.update(component_reasons)
        diagnostics.extend(component_diagnostics)

    return (
        tuple(shifts[index] for index in range(len(atoms))),
        tuple(reasons[index] for index in range(len(atoms))),
        tuple(diagnostics),
    )


def _shift_token(shift: LatticeShift) -> str:
    return "_".join(str(value) for value in shift)


def _canonical_source_edge(edge: _GraphEdge) -> tuple[int, int, LatticeShift]:
    forward = (edge.i, edge.j, edge.offset)
    reverse = (edge.j, edge.i, _negate_shift(edge.offset))
    return min(forward, reverse)


def _replica_translations(settings: CellPeriodicSettings) -> tuple[LatticeShift, ...]:
    return tuple(
        product(
            range(settings.a.start, settings.a.end),
            range(settings.b.start, settings.b.end),
            range(settings.c.start, settings.c.end),
        )
    )


def build_periodic_display(
    atoms: Atoms,
    matched_bonds: Iterable[ResolvedBond],
    settings: CellPeriodicSettings,
    *,
    topology_bonds: Iterable[ResolvedBond] | None = None,
) -> PeriodicDisplay:
    """由拓扑边确定展开，由全部匹配键构建显示实例。"""
    normalized = normalize_periodic_settings(atoms, settings)
    display_edges = _validate_graph_edges(len(atoms), matched_bonds)
    topology_edges = (
        display_edges
        if topology_bonds is None
        else _validate_graph_edges(len(atoms), topology_bonds)
    )
    adjacency = _build_adjacency(len(atoms), topology_edges)
    base_shifts, base_reasons, diagnostics = _base_image_shifts(
        atoms,
        topology_edges,
        adjacency,
        normalized.unwrap_bonded_groups,
    )
    translations = _replica_translations(normalized)
    cell = np.asarray(atoms.cell, dtype=float)

    atom_instances = []
    atom_by_key: dict[AtomInstanceKey, AtomDisplayInstance] = {}
    for replica_translation in translations:
        for source_atom_index in range(len(atoms)):
            image_shift = _add_shifts(
                base_shifts[source_atom_index],
                replica_translation,
            )
            position = np.asarray(atoms.positions[source_atom_index], dtype=float) + np.dot(
                np.asarray(image_shift, dtype=int),
                cell,
            )
            instance = AtomDisplayInstance(
                source_atom_index=source_atom_index,
                replica_translation=replica_translation,
                image_shift=image_shift,
                position=position,
                instance_id=(
                    f"atom_{source_atom_index:04d}__image_{_shift_token(image_shift)}"
                ),
            )
            key = (source_atom_index, replica_translation)
            atom_instances.append(instance)
            atom_by_key[key] = instance

    translation_set = frozenset(translations)
    seen_bonds: set[
        tuple[str, tuple[AtomInstanceKey, AtomInstanceKey]]
    ] = set()
    bond_instances = []
    for edge in display_edges:
        bond = edge.bond
        for replica_translation in translations:
            j_replica = _add_shifts(
                replica_translation,
                base_shifts[edge.i],
                edge.offset,
                _negate_shift(base_shifts[edge.j]),
            )
            if j_replica not in translation_set:
                continue
            atom_i_key = (edge.i, replica_translation)
            atom_j_key = (edge.j, j_replica)
            canonical_endpoints = tuple(sorted((atom_i_key, atom_j_key)))
            dedupe_key = (bond.bond_id, canonical_endpoints)
            if dedupe_key in seen_bonds:
                continue
            seen_bonds.add(dedupe_key)
            atom_i = atom_by_key[atom_i_key]
            atom_j = atom_by_key[atom_j_key]
            source_i, source_j, source_offset = _canonical_source_edge(edge)
            displayed_endpoints = tuple(
                sorted(
                    (
                        (atom_i.source_atom_index, atom_i.image_shift),
                        (atom_j.source_atom_index, atom_j.image_shift),
                    )
                )
            )
            first_displayed, second_displayed = displayed_endpoints
            bond_instances.append(
                BondDisplayInstance(
                    source_bond=bond,
                    atom_i_key=atom_i_key,
                    atom_j_key=atom_j_key,
                    bond_instance_id=(
                        f"{bond.bond_id}__source_{source_i}_{source_j}"
                        f"_offset_{_shift_token(source_offset)}"
                        f"__display_{first_displayed[0]}_"
                        f"{_shift_token(first_displayed[1])}"
                        f"__{second_displayed[0]}_"
                        f"{_shift_token(second_displayed[1])}"
                    ),
                )
            )

    return PeriodicDisplay(
        base_image_shifts=base_shifts,
        base_shift_reasons=base_reasons,
        replica_translations=translations,
        atom_instances=tuple(atom_instances),
        bond_instances=tuple(bond_instances),
        diagnostics=diagnostics,
        atom_by_key=MappingProxyType(atom_by_key),
    )
