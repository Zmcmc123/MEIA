"""Reusable structure topology for style-only MEIA reruns."""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

import numpy as np
from ase import Atoms

from .atom_styles import (
    AtomHydrogenBondOverride,
    HiddenAtom,
    atom_color_override_mapping,
    color_strength_mapping,
    validate_atom_selection_settings,
)
from .bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondResolution,
    BondSettings,
    BondStyle,
    ResolvedBond,
)
from .config import RenderConfig
from .hydrogen_bonds import (
    DisplayHydrogenBond,
    HydrogenBondCandidate,
    HydrogenBondSettings,
    instantiate_periodic_hydrogen_bonds,
    resolve_hydrogen_bond_candidates,
)
from .periodic_display import (
    CellPeriodicSettings,
    PeriodicDisplay,
    PeriodicRange,
    build_periodic_display,
    normalize_periodic_settings,
)
from .size_profiles import (
    RadiusMode,
    resolve_active_bond_width,
    resolve_display_radii as resolve_profile_display_radii,
)
from .view_state import camera_to_rotation_matrix
from .visual_state import (
    BondModuleSettings,
    PairRuleDefaults,
    RenderContext,
    VisualizationState,
    _resolve_context_bonds,
)


def _structure_digest(atoms: Atoms) -> str:
    if not isinstance(atoms, Atoms):
        raise TypeError("render topology must be bound to ASE Atoms")
    digest = sha256()
    for values, dtype in (
        (atoms.get_atomic_numbers(), "<i4"),
        (atoms.get_positions(), "<f8"),
        (atoms.cell.array, "<f8"),
        (atoms.pbc, "u1"),
    ):
        array = np.ascontiguousarray(values, dtype=dtype)
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class TopologyKey:
    """Stable identity of every input that can alter rendered connectivity."""

    structure_id: str
    structure_digest: str
    bond_defaults: PairRuleDefaults
    draw_bonds: bool
    pair_rules: tuple[BondPairRule, ...]
    bond_overrides: tuple[AtomBondOverride, ...]
    periodic_ranges: tuple[PeriodicRange, PeriodicRange, PeriodicRange]
    unwrap_bonded_groups: bool
    hidden_atoms: tuple[HiddenAtom, ...]
    hydrogen_bond_settings: HydrogenBondSettings
    hydrogen_bond_overrides: tuple[AtomHydrogenBondOverride, ...]


@dataclass(frozen=True)
class RenderTopology:
    """Expensive bond, periodic-display, and hydrogen-candidate results."""

    key: TopologyKey
    bonds: BondModuleSettings
    cell_periodic: CellPeriodicSettings
    bond_resolution: BondResolution
    periodic_topology_bonds: tuple[ResolvedBond, ...]
    periodic_display: PeriodicDisplay
    hydrogen_bond_candidates: tuple[HydrogenBondCandidate, ...]


@dataclass(frozen=True)
class TopologyCacheEntry:
    """One session-local topology cache entry."""

    key: TopologyKey
    topology: RenderTopology

    def __post_init__(self) -> None:
        if not isinstance(self.key, TopologyKey):
            raise TypeError("topology cache key must be TopologyKey")
        if not isinstance(self.topology, RenderTopology):
            raise TypeError("topology cache value must be RenderTopology")
        if self.topology.key != self.key:
            raise ValueError("topology cache entry key does not match its value")


def topology_key(
    atoms: Atoms,
    state: VisualizationState,
    *,
    structure_id: str | None = None,
) -> TopologyKey:
    """Build a deterministic key while excluding colors, sizes, and camera state."""
    if not isinstance(state, VisualizationState):
        raise TypeError("visual state must be VisualizationState")
    digest = _structure_digest(atoms)
    identity = digest if structure_id is None else structure_id
    if not isinstance(identity, str) or not identity.strip():
        raise ValueError("structure_id must be a non-empty string")
    normalized = normalize_periodic_settings(atoms, state.style.cell_periodic)
    bonds = state.style.bonds
    selection = state.atom_selection
    return TopologyKey(
        structure_id=identity,
        structure_digest=digest,
        bond_defaults=bonds.defaults,
        draw_bonds=bonds.draw_bonds,
        pair_rules=tuple(bonds.pair_rules),
        bond_overrides=tuple(selection.bond_overrides),
        periodic_ranges=(normalized.a, normalized.b, normalized.c),
        unwrap_bonded_groups=normalized.unwrap_bonded_groups,
        hidden_atoms=tuple(selection.hidden_atoms),
        hydrogen_bond_settings=bonds.hydrogen_bonds,
        hydrogen_bond_overrides=tuple(selection.hydrogen_bond_overrides),
    )


def build_render_topology(
    atoms: Atoms,
    state: VisualizationState,
    *,
    structure_id: str | None = None,
) -> RenderTopology:
    """Resolve expensive connectivity once without mutating the input structure."""
    key = topology_key(atoms, state, structure_id=structure_id)
    cell_periodic = normalize_periodic_settings(atoms, state.style.cell_periodic)
    bonds, _unused_settings, bond_resolution = _resolve_context_bonds(
        atoms,
        state.style.bonds,
        state.atom_selection.bond_overrides,
        resolve_active_bond_width(state.style.size_profiles),
    )
    available_pairs = tuple(rule.pair for rule in bonds.pair_rules)
    validate_atom_selection_settings(atoms, state.atom_selection, available_pairs)
    rules_by_pair = {rule.pair: rule for rule in bonds.pair_rules}
    periodic_topology_bonds = tuple(
        bond
        for bond in bond_resolution.matched
        if rules_by_pair[bond.pair].participates_in_periodic_unwrap
    )
    periodic_display = build_periodic_display(
        atoms,
        bond_resolution.matched,
        cell_periodic,
        topology_bonds=periodic_topology_bonds,
    )
    hydrogen_settings = bonds.hydrogen_bonds
    hydrogen_bond_candidates = (
        resolve_hydrogen_bond_candidates(
            atoms,
            bond_resolution.matched,
            max_hydrogen_oxygen_distance=(
                hydrogen_settings.max_hydrogen_oxygen_distance
            ),
            min_angle_degrees=hydrogen_settings.min_angle_degrees,
        )
        if hydrogen_settings.draw
        else ()
    )
    return RenderTopology(
        key=key,
        bonds=bonds,
        cell_periodic=cell_periodic,
        bond_resolution=bond_resolution,
        periodic_topology_bonds=periodic_topology_bonds,
        periodic_display=periodic_display,
        hydrogen_bond_candidates=hydrogen_bond_candidates,
    )


def _render_config(
    atoms: Atoms,
    state: VisualizationState,
    bonds: BondModuleSettings,
    cell_periodic: CellPeriodicSettings,
) -> RenderConfig:
    available_pairs = tuple(rule.pair for rule in bonds.pair_rules)
    validate_atom_selection_settings(atoms, state.atom_selection, available_pairs)
    color_strengths = color_strength_mapping(
        state.atom_selection.color_strengths,
        state.atom_selection.default_color_strength,
    )
    symbols = atoms.get_chemical_symbols()
    display_radii = resolve_profile_display_radii(state.style.size_profiles, symbols)
    element_radii: dict[str, float] = {}
    for symbol, radius in zip(symbols, display_radii):
        radius_value = float(radius)
        previous = element_radii.setdefault(symbol, radius_value)
        if previous != radius_value:
            raise ValueError(f"inconsistent display radius resolved for {symbol}")
    active_profile = (
        state.style.size_profiles.covalent
        if state.style.size_profiles.active_mode is RadiusMode.COVALENT
        else state.style.size_profiles.uniform
    )
    return RenderConfig(
        radius_scale=active_profile.global_scale,
        resolved_element_radii_angstrom=element_radii,
        outline_width=state.style.atom_cell.outline_width,
        bond_cutoff=bonds.defaults.bond_cutoff,
        bond_width_ratio=resolve_active_bond_width(state.style.size_profiles),
        bond_stroke_color=bonds.style.stroke_color,
        bond_stroke_width=bonds.style.stroke_width,
        transparent=state.style.export.transparent,
        dpi=state.style.export.dpi,
        rotation=state.style.view.rotation,
        rotation_matrix=camera_to_rotation_matrix(state.style.view.camera),
        show_unit_cell=cell_periodic.show_unit_cell,
        custom_colors=dict(state.style.atom_cell.element_colors),
        allowed_pairs=set(available_pairs),
        atom_color_strengths=color_strengths,
        atom_default_color_strength=state.atom_selection.default_color_strength,
        atom_color_overrides=atom_color_override_mapping(
            state.atom_selection.color_overrides
        ),
    )


def compose_render_context(
    atoms: Atoms,
    state: VisualizationState,
    topology: RenderTopology,
    *,
    structure_id: str | None = None,
) -> RenderContext:
    """Apply cheap visual state to a validated topology result."""
    if not isinstance(topology, RenderTopology):
        raise TypeError("topology must be RenderTopology")
    current_key = topology_key(atoms, state, structure_id=structure_id)
    if current_key != topology.key:
        raise ValueError("render topology does not match current topology inputs")
    cell_periodic = normalize_periodic_settings(atoms, state.style.cell_periodic)
    bonds = replace(
        state.style.bonds,
        pair_rules=topology.bonds.pair_rules,
    )
    bond_width_ratio = resolve_active_bond_width(state.style.size_profiles)
    bond_settings = BondSettings(
        draw_bonds=bonds.draw_bonds,
        pair_rules=bonds.pair_rules,
        atom_overrides=state.atom_selection.bond_overrides,
        style=BondStyle(
            width_ratio=bond_width_ratio,
            stroke_width=bonds.style.stroke_width,
            stroke_color=bonds.style.stroke_color,
        ),
    )
    color_strengths = color_strength_mapping(
        state.atom_selection.color_strengths,
        state.atom_selection.default_color_strength,
    )
    hydrogen_bonds: tuple[DisplayHydrogenBond, ...] = (
        instantiate_periodic_hydrogen_bonds(
            atoms,
            topology.periodic_display,
            topology.hydrogen_bond_candidates,
            state.atom_selection,
            color_strengths,
            default_color_strength=state.atom_selection.default_color_strength,
        )
        if bonds.hydrogen_bonds.draw
        else ()
    )
    return RenderContext(
        config=_render_config(atoms, state, bonds, cell_periodic),
        bond_settings=bond_settings,
        bond_resolution=topology.bond_resolution,
        periodic_topology_bonds=topology.periodic_topology_bonds,
        periodic_display=topology.periodic_display,
        hydrogen_bonds=hydrogen_bonds,
        hidden_atom_indices=frozenset(
            item.atom_index for item in state.atom_selection.hidden_atoms
        ),
    )
