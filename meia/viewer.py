"""Plotly 3D Figure 生成与 AtomViewer 组件 seam。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_rgb

from .atom_styles import AtomSelectionSettings, apply_color_strength
from .bond_segments import clip_bond_to_spheres
from .bond_rules import (
    BondSettings,
    initialize_bond_settings,
    resolve_bonds,
)
from .components.atom_viewer import render_atom_viewer
from .config import RenderConfig
from .hydrogen_bonds import (
    HYDROGEN_BOND_3D_WIDTH,
    instantiate_periodic_hydrogen_bonds,
    resolve_hydrogen_bonds,
)
from .periodic_display import CellPeriodicSettings, build_periodic_display
from .view_state import CameraState


ATOM_3D_OPACITY = 0.95
ATOM_3D_OUTLINE_WIDTH = 1.0
ATOM_3D_OUTLINE_COLOR = "#000000"
BOND_3D_OUTLINE_WIDTH = 0.5
BOND_3D_OUTLINE_DARKEN = 0.55
BOND_3D_OUTLINE_OPACITY = 0.72

_FIGURE_MESSAGE_KEYS = frozenset(
    {
        "atoms",
        "atom_hover",
        "current_selection",
        "batch_selection",
        "bond_outline",
        "bonds",
        "hydrogen_bonds",
        "unit_cell",
    }
)


def _validate_figure_messages(messages: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(messages, Mapping):
        raise TypeError("figure_messages must be a string mapping")
    missing = sorted(_FIGURE_MESSAGE_KEYS - set(messages))
    if missing:
        raise ValueError(f"missing 3D figure messages: {missing}")
    invalid = sorted(
        key
        for key in _FIGURE_MESSAGE_KEYS
        if not isinstance(messages[key], str) or not messages[key]
    )
    if invalid:
        raise ValueError(f"invalid 3D figure messages: {invalid}")
    return {key: messages[key] for key in _FIGURE_MESSAGE_KEYS}


def _soft_bond_outline_color(color: str) -> str:
    """为 3D 半键生成同元素深色半透明轮廓，降低共线覆盖的突兀感。"""
    red, green, blue = (
        round(channel * 255 * BOND_3D_OUTLINE_DARKEN)
        for channel in to_rgb(color)
    )
    return (
        f"rgba({red},{green},{blue},{BOND_3D_OUTLINE_OPACITY:.2f})"
    )


def create_3d_figure(
    atoms: Any,
    config: RenderConfig,
    draw_bonds: bool = True,
    camera: CameraState | None = None,
    uirevision: str | None = None,
    bond_settings: BondSettings | None = None,
    selected_atom_index: int | None = None,
    selected_atom_indices: Sequence[int] | None = None,
    render_context=None,
    *,
    figure_messages: Mapping[str, str],
) -> go.Figure:
    """创建带稳定已应用相机的 Plotly 3D 原子构型图。"""
    figure_text = _validate_figure_messages(figure_messages)
    if render_context is not None:
        config = render_context.config
        bond_settings = render_context.bond_settings
        draw_bonds = bond_settings.draw_bonds

    source_symbols = atoms.get_chemical_symbols()
    # 2D/3D、普通键和氢键裁剪共用同一组最终显示半径。
    source_radii = config.get_atom_radii(source_symbols)
    source_colors = config.get_atom_colors(source_symbols)
    source_strengths = config.get_atom_color_strengths(len(atoms))

    if render_context is None:
        settings = bond_settings or initialize_bond_settings(atoms, config)
        config = replace(config, bond_width_ratio=settings.style.width_ratio)
        resolution = resolve_bonds(atoms, settings) if draw_bonds else None
        matched_bonds = () if resolution is None else resolution.matched
        periodic_display = build_periodic_display(
            atoms,
            matched_bonds,
            CellPeriodicSettings(show_unit_cell=config.show_unit_cell),
        )
        candidates = resolve_hydrogen_bonds(atoms, matched_bonds)
        hydrogen_bonds = instantiate_periodic_hydrogen_bonds(
            atoms,
            periodic_display,
            candidates,
            AtomSelectionSettings(),
            {
                index: float(strength)
                for index, strength in enumerate(source_strengths)
            },
            default_color_strength=config.atom_default_color_strength,
        )
        hidden_atom_indices = frozenset()
    else:
        periodic_display = render_context.periodic_display
        hydrogen_bonds = render_context.hydrogen_bonds
        hidden_atom_indices = frozenset(render_context.hidden_atom_indices)
        config = replace(
            config,
            bond_width_ratio=render_context.bond_settings.style.width_ratio,
        )

    visible_instances = tuple(
        instance
        for instance in periodic_display.atom_instances
        if instance.source_atom_index not in hidden_atom_indices
    )
    source_atom_indices = [
        int(instance.source_atom_index) for instance in visible_instances
    ]
    positions = (
        np.vstack([instance.position for instance in visible_instances])
        if visible_instances
        else np.empty((0, 3), dtype=float)
    )
    symbols = [source_symbols[index] for index in source_atom_indices]
    radii = source_radii[source_atom_indices]
    colors = [source_colors[index] for index in source_atom_indices]
    source_outline_colors = [
        apply_color_strength(ATOM_3D_OUTLINE_COLOR, strength)
        for strength in source_strengths
    ]
    outline_colors = [
        source_outline_colors[index] for index in source_atom_indices
    ]
    customdata = [
        [
            int(instance.source_atom_index),
            source_symbols[instance.source_atom_index],
            [int(value) for value in instance.replica_translation],
            [int(value) for value in instance.image_shift],
        ]
        for instance in visible_instances
    ]

    if selected_atom_index is not None and not 0 <= selected_atom_index < len(atoms):
        raise ValueError(f"selected_atom_index out of range: {selected_atom_index}")
    if selected_atom_indices is None:
        highlighted = (() if selected_atom_index is None else (selected_atom_index,))
    else:
        highlighted = tuple(sorted(set(selected_atom_indices)))
    if any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(atoms)
        for index in highlighted
    ):
        raise ValueError("selected_atom_indices contains an out-of-range index")

    fig = go.Figure()
    atom_marker_sizes = radii * 15
    fig.add_trace(go.Scatter3d(
        x=positions[:, 0],
        y=positions[:, 1],
        z=positions[:, 2],
        mode="markers",
        marker=dict(
            size=atom_marker_sizes,
            color=colors,
            opacity=ATOM_3D_OPACITY,
            line=dict(
                color=outline_colors,
                width=ATOM_3D_OUTLINE_WIDTH,
            ),
        ),
        text=[
            figure_text["atom_hover"].format(
                symbol=symbol,
                index=source_index + 1,
                image_shift=",".join(
                    str(value) for value in instance.image_shift
                ),
            )
            for symbol, source_index, instance in zip(
                symbols,
                source_atom_indices,
                visible_instances,
            )
        ],
        customdata=customdata,
        meta={
            "meia_role": "atoms",
            "meia_base_marker_sizes": atom_marker_sizes.tolist(),
            "meia_source_atom_indices": source_atom_indices,
        },
        hoverinfo="text",
        name=figure_text["atoms"],
    ))

    if selected_atom_indices is not None or highlighted:
        selected_set = set(highlighted)
        selection_colors = [
            "rgba(255,213,79,0.55)"
            if source_index in selected_set
            else "rgba(0,0,0,0)"
            for source_index in source_atom_indices
        ]
        selection_sizes = [
            float(atom_marker_sizes[index])
            if source_index in selected_set
            else 0.0
            for index, source_index in enumerate(source_atom_indices)
        ]
        fig.add_trace(go.Scatter3d(
            x=positions[:, 0],
            y=positions[:, 1],
            z=positions[:, 2],
            mode="markers",
            marker=dict(
                size=selection_sizes,
                color=selection_colors,
                line=dict(color="#F9A825", width=2),
            ),
            customdata=customdata,
            meta={
                "meia_role": "selection",
                "meia_base_marker_sizes": atom_marker_sizes.tolist(),
                "meia_source_atom_indices": source_atom_indices,
            },
            hoverinfo="skip",
            name=(
                figure_text["batch_selection"]
                if len(highlighted) > 1
                else figure_text["current_selection"]
            ),
            showlegend=False,
        ))

    bond_instances = (
        tuple(
            instance
            for instance in periodic_display.bond_instances
            if instance.source_bond.visible
            and instance.source_bond.i not in hidden_atom_indices
            and instance.source_bond.j not in hidden_atom_indices
        )
        if draw_bonds
        else ()
    )
    if bond_instances:
        segments_by_color: dict[
            str,
            dict[str, list[float | None] | list[list[object] | None]],
        ] = {}

        def append_half(
            color: str,
            start: np.ndarray,
            end: np.ndarray,
            identity: list[object],
        ) -> None:
            coordinates = segments_by_color.setdefault(
                color,
                {"x": [], "y": [], "z": [], "customdata": []},
            )
            coordinates["x"].extend([float(start[0]), float(end[0]), None])
            coordinates["y"].extend([float(start[1]), float(end[1]), None])
            coordinates["z"].extend([float(start[2]), float(end[2]), None])
            coordinates["customdata"].extend([identity, identity, None])

        for bond_instance in bond_instances:
            bond = bond_instance.source_bond
            atom_i = periodic_display.atom_by_key[bond_instance.atom_i_key]
            atom_j = periodic_display.atom_by_key[bond_instance.atom_j_key]
            segment = clip_bond_to_spheres(
                atom_i.position,
                atom_j.position,
                source_radii[bond.i],
                source_radii[bond.j],
            )
            if segment is None:
                continue
            identity = [
                bond_instance.bond_instance_id,
                int(bond.i),
                int(bond.j),
                [int(value) for value in atom_i.image_shift],
                [int(value) for value in atom_j.image_shift],
            ]
            append_half(
                source_colors[bond.i],
                segment.start,
                segment.midpoint,
                identity,
            )
            append_half(
                source_colors[bond.j],
                segment.midpoint,
                segment.end,
                identity,
            )

        fill_width = max(1.0, config.bond_width_ratio * 8.0)
        outline_width = fill_width + 2.0 * BOND_3D_OUTLINE_WIDTH
        for color, coordinates in segments_by_color.items():
            fig.add_trace(go.Scatter3d(
                x=coordinates["x"],
                y=coordinates["y"],
                z=coordinates["z"],
                mode="lines",
                line=dict(
                    color=_soft_bond_outline_color(color),
                    width=outline_width,
                ),
                name=figure_text["bond_outline"],
                legendgroup="bond-outlines",
                showlegend=False,
                customdata=coordinates["customdata"],
                hoverinfo="skip",
                meta={
                    "meia_role": "bond_outlines",
                    "meia_base_line_width": outline_width,
                },
            ))
        for index, (color, coordinates) in enumerate(segments_by_color.items()):
            fig.add_trace(go.Scatter3d(
                x=coordinates["x"],
                y=coordinates["y"],
                z=coordinates["z"],
                mode="lines",
                line=dict(color=color, width=fill_width),
                name=figure_text["bonds"],
                legendgroup="bonds",
                showlegend=index == 0,
                customdata=coordinates["customdata"],
                hoverinfo="skip",
                meta={
                    "meia_role": "bonds",
                    "meia_base_line_width": fill_width,
                },
            ))

    if hydrogen_bonds:
        hydrogen_by_color: dict[
            str,
            dict[str, list[float | None] | list[list[object] | None]],
        ] = {}
        for hydrogen_bond in hydrogen_bonds:
            candidate = hydrogen_bond.candidate
            participants = (
                candidate.donor_oxygen,
                candidate.hydrogen,
                candidate.acceptor_oxygen,
            )
            if not hydrogen_bond.visible or any(
                index in hidden_atom_indices for index in participants
            ):
                continue
            donor_instance = periodic_display.atom_by_key[
                hydrogen_bond.donor_oxygen_key
            ]
            hydrogen_instance = periodic_display.atom_by_key[
                hydrogen_bond.hydrogen_key
            ]
            acceptor_instance = periodic_display.atom_by_key[
                hydrogen_bond.acceptor_oxygen_key
            ]
            segment = clip_bond_to_spheres(
                hydrogen_instance.position,
                acceptor_instance.position,
                source_radii[candidate.hydrogen],
                source_radii[candidate.acceptor_oxygen],
            )
            if segment is None:
                continue
            identity = [
                int(candidate.donor_oxygen),
                int(candidate.hydrogen),
                int(candidate.acceptor_oxygen),
                hydrogen_bond.instance_id,
                [int(value) for value in donor_instance.image_shift],
                [int(value) for value in hydrogen_instance.image_shift],
                [int(value) for value in acceptor_instance.image_shift],
            ]
            coordinates = hydrogen_by_color.setdefault(
                hydrogen_bond.color,
                {"x": [], "y": [], "z": [], "customdata": []},
            )
            coordinates["x"].extend(
                [float(segment.start[0]), float(segment.end[0]), None]
            )
            coordinates["y"].extend(
                [float(segment.start[1]), float(segment.end[1]), None]
            )
            coordinates["z"].extend(
                [float(segment.start[2]), float(segment.end[2]), None]
            )
            coordinates["customdata"].extend([identity, identity, None])
        for index, (color, coordinates) in enumerate(hydrogen_by_color.items()):
            fig.add_trace(go.Scatter3d(
                x=coordinates["x"],
                y=coordinates["y"],
                z=coordinates["z"],
                mode="lines",
                line=dict(
                    color=color,
                    width=HYDROGEN_BOND_3D_WIDTH,
                    dash="dash",
                ),
                customdata=coordinates["customdata"],
                meta={
                    "meia_role": "hydrogen_bonds",
                    "meia_base_line_width": HYDROGEN_BOND_3D_WIDTH,
                },
                name=figure_text["hydrogen_bonds"],
                legendgroup="hydrogen-bonds",
                showlegend=index == 0,
                hoverinfo="skip",
            ))

    cell = atoms.get_cell()
    if config.show_unit_cell > 0 and cell is not None and (cell != 0).any():
        vertices = np.array([
            [0, 0, 0],
            cell[0],
            cell[1],
            cell[0] + cell[1],
            cell[2],
            cell[0] + cell[2],
            cell[1] + cell[2],
            cell[0] + cell[1] + cell[2],
        ])
        edges = [
            (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
            (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
        ]
        cell_x, cell_y, cell_z = [], [], []
        for start, end in edges:
            cell_x.extend([vertices[start, 0], vertices[end, 0], None])
            cell_y.extend([vertices[start, 1], vertices[end, 1], None])
            cell_z.extend([vertices[start, 2], vertices[end, 2], None])

        fig.add_trace(go.Scatter3d(
            x=cell_x,
            y=cell_y,
            z=cell_z,
            mode="lines",
            line=dict(color="gray", width=2, dash="dash"),
            name=figure_text["unit_cell"],
            hoverinfo="skip",
        ))

    active_camera = camera or CameraState()
    background = "rgba(0,0,0,0)" if config.transparent else "#FFFFFF"
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            camera=active_camera.to_plotly_dict(),
            uirevision=uirevision,
            bgcolor=background,
        ),
        paper_bgcolor=background,
        plot_bgcolor=background,
        margin=dict(l=0, r=0, t=0, b=0),
        height=450,
        showlegend=True,
        legend=dict(x=0, y=1),
    )
    return fig


def atom_viewer(
    *,
    figure: go.Figure,
    structure_id: str,
    view_revision: str,
    applied_camera: CameraState,
    locale,
    messages: Mapping[str, str],
    selected_atom_index: int | None = None,
    key: str,
    axis_cameras: Mapping[str, CameraState] | None = None,
    style_dirty: bool = False,
    selected_atom_indices: Sequence[int] | None = None,
    batch_selection_enabled: bool = False,
) -> Any:
    """渲染项目自有 Viewer，并返回最新组件事件。"""
    return render_atom_viewer(
        figure=figure,
        structure_id=structure_id,
        view_revision=view_revision,
        applied_camera=applied_camera,
        locale=locale,
        messages=messages,
        axis_cameras=axis_cameras,
        selected_atom_index=selected_atom_index,
        selected_atom_indices=selected_atom_indices,
        batch_selection_enabled=batch_selection_enabled,
        style_dirty=style_dirty,
        key=key,
    )
