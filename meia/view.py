"""
3D 交互预览与相机视角转换模块。

提供 Plotly 相机参数 → 旋转矩阵的转换，
以及从 Atoms 渲染 2D 图的便捷接口。
"""

from __future__ import annotations

from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ase import Atoms

from .atom_styles import AtomSelectionSettings
from .config import RenderConfig
from .projection import project_periodic_display
from .bond_rules import BondSettings, initialize_bond_settings, resolve_bonds
from .geometry import compute_bond_geometries
from .hydrogen_bonds import (
    compute_hydrogen_bond_geometries,
    instantiate_periodic_hydrogen_bonds,
    resolve_hydrogen_bond_candidates,
)
from .renderer import render
from .periodic_display import CellPeriodicSettings, build_periodic_display
from .view_state import camera_to_rotation_matrix


def render_2d(
    atoms: Atoms,
    config: RenderConfig,
    draw_bonds: bool = True,
    bond_settings: BondSettings | None = None,
    render_context=None,
    fig=None,
    ax=None,
) -> plt.Figure:
    """用渲染引擎生成 2D 扁平化图。"""
    if render_context is not None:
        config = render_context.config
        bond_settings = render_context.bond_settings
        draw_bonds = bond_settings.draw_bonds
        resolution = render_context.bond_resolution
        periodic_display = render_context.periodic_display
        hydrogen_bonds = render_context.hydrogen_bonds
        hidden_atom_indices = render_context.hidden_atom_indices
    else:
        settings = bond_settings or initialize_bond_settings(atoms, config)
        draw_bonds = draw_bonds and settings.draw_bonds
        if settings.draw_bonds != draw_bonds:
            settings = replace(settings, draw_bonds=False)
        bond_settings = settings
        resolution = resolve_bonds(atoms, settings)
        periodic_display = build_periodic_display(
            atoms,
            resolution.matched,
            CellPeriodicSettings(show_unit_cell=config.show_unit_cell),
        )
        hydrogen_bonds = instantiate_periodic_hydrogen_bonds(
            atoms,
            periodic_display,
            resolve_hydrogen_bond_candidates(atoms, resolution.matched),
            AtomSelectionSettings(),
            config.atom_color_strengths,
        )
        hidden_atom_indices = frozenset()

    effective_config = replace(
        config,
        bond_width_ratio=bond_settings.style.width_ratio,
        bond_stroke_width=bond_settings.style.stroke_width,
        bond_stroke_color=bond_settings.style.stroke_color,
    )
    proj = project_periodic_display(
        atoms,
        periodic_display,
        effective_config,
        hidden_atom_indices,
    )
    bond_geoms = compute_bond_geometries(
        periodic_display.bond_instances,
        proj,
        effective_config,
    )
    hydrogen_bond_geoms = compute_hydrogen_bond_geometries(
        hydrogen_bonds,
        proj,
    )
    fig = render(
        proj,
        bond_geoms,
        effective_config,
        fig=fig,
        ax=ax,
        hydrogen_bond_geoms=hydrogen_bond_geoms,
    )
    return fig
