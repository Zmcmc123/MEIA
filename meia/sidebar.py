"""Streamlit 侧边栏表单适配层：只构造类型化候选值，不直接写渲染状态。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

import plotly.graph_objects as go
import streamlit as st
from ase import Atoms
from ase.data import atomic_numbers

from .presets import StylePreset
from .atom_styles import (
    AtomSelectionOperation,
    AtomSelectionSettings,
    apply_atom_selection_operation,
    emphasize_subject,
    parse_atom_index_expression,
    replace_selected_indices,
    validate_atom_selection_settings,
)
from .bond_rules import (
    BondPairRule,
    BondRuleError,
    BondSettings,
    BondStrokeStyle,
    BondStyle,
    default_pair_max_distance,
    normalize_element_pair,
    resolve_bonds,
    validate_bond_settings,
    OverrideVisibility,
)
from .hydrogen_bonds import HydrogenBondSettings
from .i18n import I18n, Locale
from .periodic_display import (
    CellPeriodicSettings,
    PeriodicRange,
    estimate_periodic_atom_instances,
    normalize_periodic_settings,
)
from .selection_paging import (
    ATOM_SELECTION_PAGE_SIZE,
    LARGE_SELECTION_THRESHOLD,
    apply_page_selection,
    selection_page,
)
from .visual_state import (
    AtomCellSettings,
    BondModuleSettings,
    ExportSettings,
    VisualizationState,
    merge_portable_style_for_structure,
)
from .size_profiles import (
    RadiusMode,
    SizeProfileSettings,
    apply_size_profile_edits,
    replace_active_bond_width,
    resolve_active_bond_width,
    resolve_display_radii,
)


VISUAL_STATE_KEY = "meia_visual_state"
ATOM_CELL_DRAFT_TOKEN_KEY = "meia_atom_cell_applied_state_token"
ATOM_SELECTION_DRAFT_REVISION_KEY = "meia_selection_draft_revision"

CELL_LABEL_KEYS = {
    0: "periodic.cell.hidden",
    1: "periodic.cell.edges",
    2: "periodic.cell.layered",
}

PAIR_VISIBILITY_OPTIONS = (
    ("unchanged", None),
    ("inherit", OverrideVisibility.INHERIT),
    ("show", OverrideVisibility.SHOW),
    ("hide", OverrideVisibility.HIDE),
)
PAIR_VISIBILITY_VALUES = dict(PAIR_VISIBILITY_OPTIONS)
COLOR_ACTIONS = ("set", "inherit")
ATOM_VISIBILITY_ACTIONS = ("hide", "show")


def _resolved_i18n(i18n: I18n | None) -> I18n:
    return i18n if i18n is not None else I18n(Locale.ZH_CN)


@dataclass(frozen=True)
class AtomFormSubmission:
    """原子表单一次提交的原子样式与尺寸档案候选。"""

    atom_cell: AtomCellSettings
    size_profiles: SizeProfileSettings


@dataclass(frozen=True)
class BondFormSubmission:
    """化学键表单一次提交的键设置与当前尺寸档案候选。"""

    bonds: BondModuleSettings
    size_profiles: SizeProfileSettings


def atom_selection_draft_widget_key(base: str, revision: int) -> str:
    """为选择表单草稿生成可在还原后换代的稳定控件 key。"""
    if revision <= 0:
        return base
    return f"{base}__reset_{revision}"


def _atom_cell_state_token(
    settings: AtomCellSettings,
    size_profiles: SizeProfileSettings,
) -> tuple[object, ...]:
    """对已应用原子模块生成稳定标识，用于区分草稿与新状态。"""
    return (
        size_profiles.active_mode.value,
        size_profiles.covalent.global_scale,
        size_profiles.covalent.bond_width_ratio,
        tuple(sorted(size_profiles.covalent.reference_overrides_angstrom.items())),
        size_profiles.uniform.global_scale,
        size_profiles.uniform.reference_radius_angstrom,
        size_profiles.uniform.bond_width_ratio,
        tuple(sorted(size_profiles.uniform.reference_overrides_angstrom.items())),
        settings.outline_width,
        tuple(sorted(settings.element_colors.items())),
    )


def _synchronize_atom_cell_drafts(
    current: AtomCellSettings,
    size_profiles: SizeProfileSettings,
) -> None:
    """已应用状态变更后，在创建控件前丢弃上一版草稿。"""
    token = _atom_cell_state_token(current, size_profiles)
    previous = st.session_state.get(ATOM_CELL_DRAFT_TOKEN_KEY)
    if previous is not None and previous != token:
        for key in tuple(st.session_state):
            if key.startswith("meia_atom_cell_") and key != ATOM_CELL_DRAFT_TOKEN_KEY:
                st.session_state.pop(key, None)
    st.session_state[ATOM_CELL_DRAFT_TOKEN_KEY] = token


def _bounds_including_values(
    preferred_minimum: float,
    preferred_maximum: float,
    *values: object,
) -> tuple[float, float]:
    """保留常用编辑范围，并承接已应用值及当前控件草稿。"""
    finite_values: list[float] = []
    for value in values:
        if isinstance(value, bool):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            finite_values.append(numeric)
    return (
        min([preferred_minimum, *finite_values]),
        max([preferred_maximum, *finite_values]),
    )


def render_atom_cell_form(
    current: AtomCellSettings,
    size_profiles: SizeProfileSettings,
    atoms: Atoms,
    default_colors: Mapping[str, str],
    i18n: I18n | None = None,
) -> AtomFormSubmission | None:
    """缓冲双模式半径设置，仅在明确提交时返回完整原子模块状态。"""
    i18n = _resolved_i18n(i18n)
    _synchronize_atom_cell_drafts(current, size_profiles)
    elements = sorted(
        set(atoms.get_chemical_symbols()), key=lambda symbol: atomic_numbers[symbol]
    )
    modes = tuple(mode.value for mode in RadiusMode)
    draft_mode = RadiusMode(st.selectbox(
        i18n.text("atom.radius_mode"),
        modes,
        index=modes.index(size_profiles.active_mode.value),
        format_func=lambda mode: i18n.text(f"atom.radius_mode.{mode}"),
        key="meia_atom_cell_radius_mode",
    ))
    st.caption(i18n.text("atom.radius_mode.help"))

    def radius_defaults(mode: RadiusMode, scale: float, uniform_base: float):
        draft_settings = replace(size_profiles, active_mode=mode)
        if mode is RadiusMode.COVALENT:
            draft_settings = replace(
                draft_settings,
                covalent=replace(draft_settings.covalent, global_scale=scale),
            )
        else:
            draft_settings = replace(
                draft_settings,
                uniform=replace(
                    draft_settings.uniform,
                    global_scale=scale,
                    reference_radius_angstrom=uniform_base,
                ),
            )
        return dict(
            zip(
                elements,
                resolve_display_radii(
                    draft_settings,
                    elements,
                ),
            )
        )

    with st.form("meia_atom_cell_form", clear_on_submit=False):
        active_key_prefix = (
            "covalent" if draft_mode is RadiusMode.COVALENT else "uniform"
        )
        draft_profile = (
            size_profiles.covalent
            if draft_mode is RadiusMode.COVALENT
            else size_profiles.uniform
        )
        applied_global_scale = float(draft_profile.global_scale)
        scale_widget_key = (
            f"meia_atom_cell_{active_key_prefix}_global_scale"
        )
        scale_minimum, scale_maximum = _bounds_including_values(
            0.1,
            1.5,
            applied_global_scale,
            st.session_state.get(scale_widget_key),
        )
        global_scale = st.slider(
            i18n.text("atom.global_radius_scale"),
            scale_minimum,
            scale_maximum,
            applied_global_scale,
            0.05,
            key=scale_widget_key,
        )
        uniform_reference = size_profiles.uniform.reference_radius_angstrom
        if draft_mode is RadiusMode.UNIFORM:
            uniform_widget_key = "meia_atom_cell_uniform_reference_radius"
            uniform_minimum, uniform_maximum = _bounds_including_values(
                0.1,
                5.0,
                uniform_reference,
                st.session_state.get(uniform_widget_key),
            )
            uniform_reference = st.number_input(
                i18n.text("atom.uniform_base_radius"),
                min_value=uniform_minimum,
                max_value=uniform_maximum,
                value=float(uniform_reference),
                step=0.05,
                format="%.2f",
                key=uniform_widget_key,
            )

        active_defaults = radius_defaults(
            draft_mode, global_scale, uniform_reference
        )
        edited_radii: dict[str, float] = {}
        for symbol in elements:
            widget_key = (
                f"meia_atom_cell_{active_key_prefix}"
                f"_display_radius_{symbol}"
            )
            baseline_key = f"{widget_key}_baseline"
            applied_display_radius = float(active_defaults[symbol])
            previous_baseline = st.session_state.get(baseline_key)
            if (
                widget_key in st.session_state
                and previous_baseline is not None
                and math.isclose(
                    float(st.session_state[widget_key]),
                    float(previous_baseline),
                    rel_tol=0.0,
                    abs_tol=1.0e-9,
                )
            ):
                st.session_state[widget_key] = applied_display_radius
            radius_minimum, radius_maximum = _bounds_including_values(
                0.01,
                5.0,
                applied_display_radius,
                st.session_state.get(widget_key),
            )
            selected = st.number_input(
                i18n.text("atom.element_display_radius", symbol=symbol),
                min_value=radius_minimum,
                max_value=radius_maximum,
                value=applied_display_radius,
                step=0.01,
                format="%.2f",
                key=widget_key,
            )
            selected_value = float(selected)
            st.session_state[baseline_key] = applied_display_radius
            if (
                not math.isfinite(selected_value)
                or selected_value <= 0
                or abs(selected_value - applied_display_radius) > 1.0e-9
            ):
                edited_radii[symbol] = selected

        outline_width = st.slider(
            i18n.text("atom.outline_width"),
            0.0,
            2.0,
            float(current.outline_width),
            0.1,
            key="meia_atom_cell_outline_width",
        )
        selected_colors = {
            symbol: st.color_picker(
                i18n.text("atom.element_color", symbol=symbol),
                current.element_colors[symbol],
                key=f"meia_atom_cell_color_{symbol}",
            )
            for symbol in elements
        }
        submitted = st.form_submit_button(
            i18n.text("atom.apply"), type="primary"
        )
        restore_colors = st.form_submit_button(i18n.text("atom.restore_colors"))

    if not submitted and not restore_colors:
        return None
    try:
        if restore_colors:
            return AtomFormSubmission(
                atom_cell=replace(current, element_colors=dict(default_colors)),
                size_profiles=size_profiles,
            )
        updated_profiles = apply_size_profile_edits(
            size_profiles,
            mode=draft_mode,
            global_scale=global_scale,
            uniform_reference_radius_angstrom=uniform_reference,
            submitted_display_radii_angstrom=edited_radii,
        )
        colors = dict(current.element_colors)
        colors.update(selected_colors)
        return AtomFormSubmission(
            atom_cell=replace(
                current,
                outline_width=outline_width,
                element_colors=colors,
            ),
            size_profiles=updated_profiles,
        )
    except (TypeError, ValueError) as exc:
        st.error(i18n.error_text(exc, "atom.apply_failed"))
        return None

def render_cell_periodic_form(
    current: CellPeriodicSettings,
    atoms: Atoms,
    i18n: I18n | None = None,
) -> CellPeriodicSettings | None:
    """缓冲晶胞图层、成键组展开和周期范围，提交时原子化应用。"""
    i18n = _resolved_i18n(i18n)
    if not isinstance(current, CellPeriodicSettings):
        raise TypeError("cell periodic settings must be CellPeriodicSettings")
    if not isinstance(atoms, Atoms):
        raise TypeError("cell periodic settings must be bound to ASE Atoms")

    axes = (("a", current.a), ("b", current.b), ("c", current.c))
    with st.form("meia_cell_periodic_form", clear_on_submit=False):
        show_unit_cell = st.selectbox(
            i18n.text("periodic.cell_display"),
            [2, 1, 0],
            index=[2, 1, 0].index(current.show_unit_cell),
            format_func=lambda value: i18n.text(CELL_LABEL_KEYS[value]),
            key="meia_cell_periodic_show_unit_cell",
        )
        unwrap_bonded_groups = st.checkbox(
            i18n.text("periodic.unwrap_bonded"),
            value=current.unwrap_bonded_groups,
            key="meia_cell_periodic_unwrap_bonded_groups",
        )
        range_values: dict[str, tuple[int, int]] = {}
        for axis_index, (axis, axis_range) in enumerate(axes):
            periodic = bool(atoms.pbc[axis_index])
            start = st.number_input(
                i18n.text("periodic.axis_start", axis=axis),
                value=int(axis_range.start if periodic else 0),
                step=1,
                format="%d",
                disabled=not periodic,
                key=f"meia_cell_periodic_{axis}_start",
            )
            end = st.number_input(
                i18n.text("periodic.axis_end", axis=axis),
                value=int(axis_range.end if periodic else 1),
                step=1,
                format="%d",
                disabled=not periodic,
                key=f"meia_cell_periodic_{axis}_end",
            )
            range_values[axis] = (int(start), int(end))
            if not periodic:
                st.caption(i18n.text("periodic.non_pbc", axis=axis))

        candidate = None
        candidate_error = None
        displayed_count = None
        displayed_period_counts = None
        try:
            raw_candidate = CellPeriodicSettings(
                show_unit_cell=int(show_unit_cell),
                unwrap_bonded_groups=unwrap_bonded_groups,
                a=PeriodicRange(*range_values["a"]),
                b=PeriodicRange(*range_values["b"]),
                c=PeriodicRange(*range_values["c"]),
            )
            displayed_period_counts = tuple(
                axis_range.end - axis_range.start if bool(atoms.pbc[index]) else 1
                for index, axis_range in enumerate(
                    (raw_candidate.a, raw_candidate.b, raw_candidate.c)
                )
            )
            displayed_count = estimate_periodic_atom_instances(
                atoms,
                raw_candidate,
            )
            candidate = normalize_periodic_settings(
                atoms,
                raw_candidate,
            )
            displayed_count = estimate_periodic_atom_instances(atoms, candidate)
        except (TypeError, ValueError) as exc:
            candidate_error = exc
        if displayed_period_counts is not None:
            a_count, b_count, c_count = displayed_period_counts
            st.caption(
                i18n.text(
                    "periodic.current_counts",
                    a=a_count,
                    b=b_count,
                    c=c_count,
                )
            )
        if displayed_count is not None:
            st.caption(i18n.text("periodic.estimated_atoms", count=f"{displayed_count:,}"))
        else:
            st.caption(i18n.text("periodic.estimate_unavailable"))
        submitted = st.form_submit_button(
            i18n.text("periodic.apply"),
            type="primary",
        )

    if not submitted:
        return None
    if candidate_error is not None:
        st.error(i18n.error_text(candidate_error, "periodic.apply_failed"))
        return None
    return candidate


def render_export_form(
    current: ExportSettings,
    i18n: I18n | None = None,
) -> ExportSettings | None:
    """缓冲导出格式、DPI 和背景，仅在提交后返回。"""
    i18n = _resolved_i18n(i18n)
    formats = ["SVG", "PNG", "PDF"]
    with st.form("meia_export_form", clear_on_submit=False):
        export_format = st.selectbox(
            i18n.text("export.format"),
            formats,
            index=formats.index(current.format.upper()),
            key="meia_export_form_format",
        )
        dpi = st.number_input(
            i18n.text("export.png_dpi"),
            100,
            1200,
            int(current.dpi),
            50,
            key="meia_export_form_dpi",
        )
        transparent = st.checkbox(
            i18n.text("export.transparent"),
            value=current.transparent,
            key="meia_export_form_transparent",
        )
        submitted = st.form_submit_button(i18n.text("export.apply"), type="primary")
    if not submitted:
        return None
    try:
        return ExportSettings(export_format.lower(), int(dpi), transparent)
    except (TypeError, ValueError) as exc:
        st.error(i18n.error_text(exc, "export.apply_failed"))
        return None


def _matched_bond_rule_counts(
    current: BondModuleSettings,
    atoms: Atoms,
) -> tuple[tuple[BondPairRule, ...], dict[tuple[str, str], int]]:
    """返回当前构型实际匹配到的持久规则及匹配数。"""
    try:
        counts = dict(
            resolve_bonds(
                atoms,
                BondSettings(
                    draw_bonds=current.draw_bonds,
                    pair_rules=current.pair_rules,
                    style=current.style,
                ),
            ).match_counts
        )
    except (BondRuleError, ValueError):
        counts = {}
    rules = tuple(
        rule for rule in current.pair_rules if counts.get(rule.pair, 0) > 0
    )
    return rules, counts


def matched_bond_pairs(
    current: BondModuleSettings,
    atoms: Atoms,
) -> tuple[tuple[str, str], ...]:
    """供化学键表单和原子选择共用的构型匹配元素对。"""
    rules, _counts = _matched_bond_rule_counts(current, atoms)
    return tuple(rule.pair for rule in rules)


def _bond_rule_summary(
    rules: tuple[BondPairRule, ...],
    counts: Mapping[tuple[str, str], int],
    i18n: I18n,
) -> go.Figure:
    return go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        i18n.text("bonds.table.pair"),
                        i18n.text("bonds.table.visible"),
                        i18n.text("bonds.table.periodic"),
                        i18n.text("bonds.table.distance"),
                        i18n.text("bonds.table.matches"),
                    ],
                    fill_color="#E8EEF5",
                    align="left",
                ),
                cells=dict(
                    values=[
                        [f"{rule.pair[0]}–{rule.pair[1]}" for rule in rules],
                        [
                            i18n.text(
                                "common.show" if rule.enabled else "common.hide"
                            )
                            for rule in rules
                        ],
                        [
                            i18n.text(
                                "bonds.table.participates"
                                if rule.participates_in_periodic_unwrap
                                else "bonds.table.excluded"
                            )
                            for rule in rules
                        ],
                        [
                            f"{rule.min_distance:.3f} – {rule.max_distance:.3f}"
                            for rule in rules
                        ],
                        [counts.get(rule.pair, 0) for rule in rules],
                    ],
                    align="left",
                ),
            )
        ]
    ).update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        height=max(95, 32 * len(rules) + 50),
    )


def render_bond_form(
    current: BondModuleSettings,
    size_profiles: SizeProfileSettings,
    atoms: Atoms,
    i18n: I18n | None = None,
) -> BondFormSubmission | None:
    """提交全局样式与当前构型实际匹配到的元素对设置。"""
    i18n = _resolved_i18n(i18n)
    matched_rules, match_counts = _matched_bond_rule_counts(current, atoms)
    if matched_rules:
        st.plotly_chart(
            _bond_rule_summary(matched_rules, match_counts, i18n),
            use_container_width=True,
            config={"displayModeBar": False},
        )
    else:
        st.caption(i18n.text("bonds.no_matches"))

    rules_by_pair = {rule.pair: rule for rule in matched_rules}
    with st.form("meia_bond_form", clear_on_submit=False):
        draw_bonds = st.checkbox(
            i18n.text("bonds.show_detected"),
            value=current.draw_bonds,
            key="meia_bond_form_draw_bonds",
        )
        width_ratio = st.slider(
            i18n.text("bonds.width_ratio"),
            0.1,
            1.0,
            resolve_active_bond_width(size_profiles),
            0.05,
            key="meia_bond_form_width_ratio",
        )
        stroke_width = st.slider(
            i18n.text("bonds.stroke_width"),
            0.0,
            2.0,
            current.style.stroke_width,
            0.05,
            key="meia_bond_form_stroke_width",
        )
        stroke_color = st.color_picker(
            i18n.text("bonds.stroke_color"),
            current.style.stroke_color,
            key="meia_bond_form_stroke_color",
        )

        rule_values = {}
        for pair, rule in rules_by_pair.items():
            label = f"{pair[0]}–{pair[1]}"
            st.markdown(f"**{label}**")
            enabled = st.checkbox(
                i18n.text("bonds.show_pair", pair=label),
                value=rule.enabled,
                key=f"meia_bond_form_{pair[0]}_{pair[1]}_enabled",
            )
            participates = st.checkbox(
                i18n.text("bonds.pair_periodic", pair=label),
                value=rule.participates_in_periodic_unwrap,
                key=f"meia_bond_form_{pair[0]}_{pair[1]}_periodic",
            )
            minimum = st.number_input(
                i18n.text("bonds.pair_minimum", pair=label),
                0.0,
                50.0,
                rule.min_distance,
                0.05,
                key=f"meia_bond_form_{pair[0]}_{pair[1]}_minimum",
            )
            maximum = st.number_input(
                i18n.text("bonds.pair_maximum", pair=label),
                0.0,
                50.0,
                rule.max_distance,
                0.05,
                key=f"meia_bond_form_{pair[0]}_{pair[1]}_maximum",
            )
            rule_values[pair] = (enabled, participates, minimum, maximum)

        st.markdown(i18n.text("hydrogen.heading"))
        draw_hydrogen_bonds = st.checkbox(
            i18n.text("hydrogen.show"),
            value=current.hydrogen_bonds.draw,
            key="meia_bond_form_hydrogen_draw",
        )
        hydrogen_maximum = st.number_input(
            i18n.text("hydrogen.maximum"),
            0.05,
            50.0,
            current.hydrogen_bonds.max_hydrogen_oxygen_distance,
            0.05,
            key="meia_bond_form_hydrogen_maximum",
        )
        hydrogen_angle = st.number_input(
            i18n.text("hydrogen.minimum_angle"),
            0.0,
            180.0,
            current.hydrogen_bonds.min_angle_degrees,
            1.0,
            key="meia_bond_form_hydrogen_angle",
        )

        elements = sorted(
            set(atoms.get_chemical_symbols()),
            key=lambda symbol: atomic_numbers[symbol],
        )
        if elements:
            add_pair_enabled = st.checkbox(
                i18n.text("bonds.add_pair"),
                value=False,
                key="meia_bond_form_add_enabled",
            )
            element_a = st.selectbox(
                i18n.text("bonds.element_a"),
                elements,
                key="meia_bond_form_element_a",
            )
            element_b = st.selectbox(
                i18n.text("bonds.element_b"),
                elements,
                key="meia_bond_form_element_b",
            )
        else:
            add_pair_enabled = False
            element_a = None
            element_b = None

        submitted = st.form_submit_button(
            i18n.text("bonds.apply"),
            type="primary",
            disabled=len(atoms) == 0,
        )

    if not submitted:
        return None
    try:
        updated_rules = {
            pair: BondPairRule(
                pair[0],
                pair[1],
                minimum,
                maximum,
                enabled=enabled,
                participates_in_periodic_unwrap=participates,
            )
            for pair, (
                enabled,
                participates,
                minimum,
                maximum,
            ) in rule_values.items()
        }
        rules = tuple(
            updated_rules.get(rule.pair, rule) for rule in current.pair_rules
        )
        if add_pair_enabled:
            added_pair = normalize_element_pair(element_a, element_b)
            if added_pair in {rule.pair for rule in rules}:
                st.warning(
                    i18n.text(
                        "bonds.pair_exists",
                        pair=f"{added_pair[0]}–{added_pair[1]}",
                    )
                )
            else:
                rules += (
                    BondPairRule(
                        added_pair[0],
                        added_pair[1],
                        0.0,
                        default_pair_max_distance(
                            added_pair[0],
                            added_pair[1],
                            bond_cutoff=current.defaults.bond_cutoff,
                            pair_distance_multipliers=(
                                current.defaults.multiplier_mapping()
                            ),
                        ),
                        enabled=True,
                        participates_in_periodic_unwrap=True,
                    ),
                )
        candidate = BondModuleSettings(
            draw_bonds=draw_bonds,
            style=BondStrokeStyle(stroke_width, stroke_color),
            defaults=current.defaults,
            pair_rules=rules,
            hydrogen_bonds=HydrogenBondSettings(
                draw_hydrogen_bonds,
                hydrogen_maximum,
                hydrogen_angle,
            ),
        )
        validate_bond_settings(
            atoms,
            BondSettings(
                draw_bonds=candidate.draw_bonds,
                pair_rules=candidate.pair_rules,
                style=BondStyle(
                    width_ratio,
                    candidate.style.stroke_width,
                    candidate.style.stroke_color,
                ),
            ),
        )
        return BondFormSubmission(
            bonds=candidate,
            size_profiles=replace_active_bond_width(size_profiles, width_ratio),
        )
    except (BondRuleError, TypeError, ValueError) as exc:
        st.error(i18n.error_text(exc, "bonds.apply_failed"))
        return None


def render_atom_selection_form(
    current: AtomSelectionSettings,
    atoms: Atoms,
    available_pairs,
    i18n: I18n | None = None,
) -> AtomSelectionSettings | None:
    """统一单选/多选入口，并一次性应用显式启用的原子操作。"""
    i18n = _resolved_i18n(i18n)
    symbols = atoms.get_chemical_symbols()
    atom_indices = list(range(len(atoms)))
    elements = sorted(set(symbols), key=lambda symbol: atomic_numbers[symbol])
    pairs = tuple(sorted(set(available_pairs)))
    hidden = {item.atom_index for item in current.hidden_atoms}
    raw_revision = st.session_state.get(ATOM_SELECTION_DRAFT_REVISION_KEY, 0)
    revision = (
        raw_revision
        if isinstance(raw_revision, int) and raw_revision >= 0
        else 0
    )

    def draft_key(base: str) -> str:
        return atom_selection_draft_widget_key(base, revision)

    large_selection = len(atoms) >= LARGE_SELECTION_THRESHOLD
    searchable: tuple[int, ...] | list[int] = ()
    page_selected: tuple[int, ...] | list[int] = ()
    page_action = "add"
    active_page = None

    with st.form(
        draft_key("meia_atom_selection_form"),
        clear_on_submit=False,
    ):
        if large_selection:
            st.caption(
                i18n.text(
                    "selection.summary_count",
                    count=len(current.selected_atom_indices),
                )
            )
            if current.selected_atom_indices:
                summary_labels = [
                    f"{symbols[index]} #{index + 1}"
                    + (
                        i18n.text("selection.hidden_suffix")
                        if index in hidden
                        else ""
                    )
                    for index in current.selected_atom_indices[:20]
                ]
                st.caption(
                    i18n.text(
                        "selection.summary_atoms",
                        atoms=", ".join(summary_labels),
                    )
                )
            page_count = (len(atoms) + ATOM_SELECTION_PAGE_SIZE - 1) // (
                ATOM_SELECTION_PAGE_SIZE
            )
            page_number = int(
                st.number_input(
                    i18n.text("selection.page_number", count=page_count),
                    min_value=1,
                    max_value=page_count,
                    value=1,
                    step=1,
                    key=draft_key("meia_atom_selection_page_number"),
                )
            )
            active_page = selection_page(len(atoms), page_number)
            page_selected = st.multiselect(
                i18n.text(
                    "selection.page_atoms",
                    count=ATOM_SELECTION_PAGE_SIZE,
                ),
                active_page.indices,
                format_func=lambda index: (
                    f"{symbols[index]} #{index + 1}"
                    + (
                        i18n.text("selection.hidden_suffix")
                        if index in hidden
                        else ""
                    )
                ),
                key=draft_key(
                    f"meia_atom_selection_page_indices_{page_number}"
                ),
            )
            page_action = st.selectbox(
                i18n.text("selection.page_action"),
                ("add", "remove"),
                format_func=lambda action: i18n.text(
                    f"selection.page_action.{action}"
                ),
                key=draft_key("meia_atom_selection_page_action"),
            )
        else:
            selection_widget_key = draft_key("meia_atom_selection_indices")
            selection_default = (
                {"default": list(current.selected_atom_indices)}
                if selection_widget_key not in st.session_state
                else {}
            )
            searchable = st.multiselect(
                i18n.text("selection.current"),
                atom_indices,
                format_func=lambda index: (
                    f"{symbols[index]} #{index + 1}"
                    + (
                        i18n.text("selection.hidden_suffix")
                        if index in hidden
                        else ""
                    )
                ),
                key=selection_widget_key,
                **selection_default,
            )
        index_expression = st.text_input(
            i18n.text("selection.by_index"),
            value="",
            placeholder=i18n.text("selection.by_index_placeholder"),
            key=draft_key("meia_atom_selection_range"),
        )
        selected_elements = st.multiselect(
            i18n.text("selection.by_element"),
            elements,
            key=draft_key("meia_atom_selection_elements"),
        )
        invert_final = st.checkbox(
            i18n.text("selection.invert"),
            value=False,
            key=draft_key("meia_atom_selection_invert"),
        )
        clear_selection = st.checkbox(
            i18n.text("selection.clear"),
            value=False,
            key=draft_key("meia_atom_selection_clear"),
        )

        st.markdown(i18n.text("selection.operations_heading"))
        change_color = st.checkbox(
            i18n.text("selection.change_color"),
            value=False,
            key=draft_key("meia_atom_selection_change_color"),
        )
        color_choice = st.selectbox(
            i18n.text("selection.color_action"),
            COLOR_ACTIONS,
            format_func=lambda action: i18n.text(
                f"selection.color_action.{action}"
            ),
            key=draft_key("meia_atom_selection_color_action"),
        )
        color = st.color_picker(
            i18n.text("selection.atom_color"),
            "#6699CC",
            key=draft_key("meia_atom_selection_color"),
        )
        change_strength = st.checkbox(
            i18n.text("selection.change_strength"),
            value=False,
            key=draft_key("meia_atom_selection_change_strength"),
        )
        strength_percent = st.slider(
            i18n.text("selection.target_strength"),
            0,
            100,
            30,
            5,
            format="%d%%",
            key=draft_key("meia_atom_selection_strength"),
        )
        emphasize_current_selection = st.checkbox(
            i18n.text("selection.emphasize_subject"),
            value=False,
            key=draft_key("meia_atom_selection_emphasize_subject"),
        )
        background_strength_percent = st.slider(
            i18n.text("selection.background_strength"),
            0,
            100,
            30,
            5,
            format="%d%%",
            key=draft_key("meia_atom_selection_background_strength"),
        )
        change_visibility = st.checkbox(
            i18n.text("selection.change_visibility"),
            value=False,
            key=draft_key("meia_atom_selection_change_visibility"),
        )
        visibility_choice = st.selectbox(
            i18n.text("selection.visibility_action"),
            ATOM_VISIBILITY_ACTIONS,
            format_func=lambda action: i18n.text(
                f"selection.visibility_action.{action}"
            ),
            key=draft_key("meia_atom_selection_visibility_action"),
        )
        hydrogen_bond_choice = st.selectbox(
            i18n.text("selection.hydrogen_rule"),
            tuple(code for code, _value in PAIR_VISIBILITY_OPTIONS),
            format_func=lambda code: i18n.text(f"selection.pair_action.{code}"),
            key=draft_key("meia_atom_selection_hydrogen_bond_visibility"),
        )
        pair_choices = {
            pair: st.selectbox(
                i18n.text(
                    "selection.bond_rule",
                    pair=f"{pair[0]}–{pair[1]}",
                ),
                tuple(code for code, _value in PAIR_VISIBILITY_OPTIONS),
                format_func=lambda code: i18n.text(
                    f"selection.pair_action.{code}"
                ),
                key=draft_key(
                    f"meia_atom_selection_pair_{pair[0]}_{pair[1]}"
                ),
            )
            for pair in pairs
        }
        submitted = st.form_submit_button(
            i18n.text("selection.apply"),
            type="primary",
            disabled=len(atoms) == 0,
        )

    if not submitted:
        return None
    try:
        if active_page is None:
            final = set(searchable)
        else:
            final = set(
                apply_page_selection(
                    current.selected_atom_indices,
                    page_selected,
                    page_action,
                    allowed_indices=active_page.indices,
                )
            )
        final.update(parse_atom_index_expression(index_expression, len(atoms)))
        final.update(
            index
            for index, symbol in enumerate(symbols)
            if symbol in set(selected_elements)
        )
        if invert_final:
            final = set(atom_indices) - final
        if clear_selection:
            final = set()

        candidate = replace_selected_indices(current, final, len(atoms))
        if change_color:
            color_action = color_choice
            operation_color = color if color_action == "set" else None
        else:
            color_action = "unchanged"
            operation_color = None
        operation = AtomSelectionOperation(
            color_action=color_action,
            color=operation_color,
            strength=strength_percent / 100.0 if change_strength else None,
            visibility_action=(
                visibility_choice if change_visibility else "unchanged"
            ),
            hydrogen_bond_visibility=PAIR_VISIBILITY_VALUES[
                hydrogen_bond_choice
            ],
            bond_visibility={
                pair: PAIR_VISIBILITY_VALUES[label]
                for pair, label in pair_choices.items()
            },
        )
        updated = apply_atom_selection_operation(
            atoms,
            candidate,
            operation,
            pairs,
        )
        if emphasize_current_selection:
            updated = emphasize_subject(
                atoms,
                updated,
                background_strength_percent / 100.0,
            )
        validate_atom_selection_settings(atoms, updated, pairs)
        return updated
    except (TypeError, ValueError) as exc:
        st.error(i18n.error_text(exc, "selection.apply_failed"))
        return None


def initialize_visual_state(
    atoms: Atoms,
    default_style: StylePreset,
) -> VisualizationState:
    """从内置 v7 风格建立结构感知的首份已应用状态。"""
    if not isinstance(default_style, StylePreset):
        raise TypeError("default style must be StylePreset")
    style = merge_portable_style_for_structure(default_style.style, atoms)
    return VisualizationState(style=style)


def store_visual_state(session_state: Any, state: VisualizationState) -> None:
    if not isinstance(state, VisualizationState):
        raise TypeError("visual state must be VisualizationState")
    session_state[VISUAL_STATE_KEY] = state


def load_visual_state(session_state: Any) -> VisualizationState:
    state = session_state[VISUAL_STATE_KEY]
    if not isinstance(state, VisualizationState):
        raise TypeError("session visual state is invalid")
    return state
