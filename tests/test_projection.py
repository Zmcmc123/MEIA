"""
投影与渲染验证测试。
"""

import sys
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle as MplCircle

from ase import Atoms
from ase.visualize.plot import plot_atoms
from ase.data import covalent_radii

from meia.config import RenderConfig
import meia.projection as projection_module
from meia.projection import project_atoms
from meia.atom_styles import AtomSelectionSettings, HiddenAtom
from meia.bond_rules import (
    BondPairRule,
    initialize_bond_settings,
    resolve_bonds,
)
from meia.periodic_display import (
    CellPeriodicSettings,
    PeriodicRange,
    build_periodic_display,
)
from meia.visual_state import (
    BondModuleSettings,
    PortableStyle,
    VisualizationState,
    resolve_render_context,
)
from meia.bonds import find_bonds
from meia.geometry import compute_bond_geometries
from meia.renderer import render
from meia.export import export_svg


def _periodic_context(atoms, *, hidden_atoms=(), show_unit_cell=0):
    state = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.0, 1.2),),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=show_unit_cell,
                a=PeriodicRange(-1, 2),
            ),
        ),
        atom_selection=AtomSelectionSettings(hidden_atoms=hidden_atoms),
    )
    return resolve_render_context(atoms, state)


def test_non_topology_ca_o_stays_visible_while_o_si_alone_sets_base_shifts():
    """显示用 Ca–O 若注入周期图，该断言会捕获到基础像偏移的变化。"""
    atoms = Atoms(
        ["Ca", "O", "Si"],
        positions=[[3.3, 0.0, 0.0], [1.5, 0.0, 0.0], [9.5, 0.0, 0.0]],
        cell=[10.0, 8.0, 8.0],
        pbc=[True, False, False],
    )
    state = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                pair_rules=(
                    BondPairRule(
                        "Ca",
                        "O",
                        0.0,
                        1.9,
                        enabled=True,
                        participates_in_periodic_unwrap=False,
                    ),
                    BondPairRule(
                        "Ca",
                        "Si",
                        0.0,
                        0.1,
                        enabled=False,
                        participates_in_periodic_unwrap=False,
                    ),
                    BondPairRule(
                        "O",
                        "Si",
                        0.0,
                        2.1,
                        enabled=True,
                        participates_in_periodic_unwrap=True,
                    ),
                )
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=0,
                a=PeriodicRange(-1, 2),
            ),
        )
    )

    context = resolve_render_context(atoms, state)

    assert not context.periodic_display.diagnostics
    assert {bond.pair for bond in context.periodic_topology_bonds} == {("O", "Si")}
    assert context.periodic_display.base_image_shifts == (
        (0, 0, 0),
        (1, 0, 0),
        (0, 0, 0),
    )
    assert any(
        instance.source_bond.pair == ("Ca", "O")
        for instance in context.periodic_display.bond_instances
    )


def test_periodic_projection_preserves_source_replica_and_image_identity():
    """把副本压回源索引会产生重复 SVG id 与错误选择身份。"""
    atoms = Atoms(
        "HHO",
        scaled_positions=[[0.01, 0.5, 0.5], [0.92, 0.5, 0.5], [0.95, 0.5, 0.5]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    context = _periodic_context(atoms)

    projection = projection_module.project_periodic_display(
        atoms,
        context.periodic_display,
        context.config,
        context.hidden_atom_indices,
    )

    assert projection.source_atom_indices.tolist() == [0, 1, 2] * 3
    assert len(set(projection.instance_keys)) == 9
    assert projection.instance_index_by_key == {
        key: index for index, key in enumerate(projection.instance_keys)
    }
    assert projection.image_shifts.shape == (9, 3)


def test_periodic_projection_filters_hidden_sources_after_display_is_built():
    """隐藏源原子只能过滤其全部副本，不能重建并移动其余拓扑。"""
    atoms = Atoms(
        "HHO",
        scaled_positions=[[0.01, 0.5, 0.5], [0.92, 0.5, 0.5], [0.95, 0.5, 0.5]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    hidden = (HiddenAtom(1, "H"),)
    context = _periodic_context(atoms, hidden_atoms=hidden)

    projection = projection_module.project_periodic_display(
        atoms,
        context.periodic_display,
        context.config,
        context.hidden_atom_indices,
    )

    assert projection.source_atom_indices.tolist() == [0, 2] * 3
    assert all(key[0] != 1 for key in projection.instance_keys)
    assert len(context.periodic_display.atom_instances) == 9


def test_all_hidden_atoms_keep_a_finite_primary_cell_canvas():
    """空原子层不得因 min/max 空数组崩溃，晶胞仍只使用主晶胞定界。"""
    atoms = Atoms(
        "HO",
        scaled_positions=[[0.02, 0.5, 0.5], [0.98, 0.5, 0.5]],
        cell=[10.0, 8.0, 6.0],
        pbc=True,
    )
    context = _periodic_context(
        atoms,
        hidden_atoms=(HiddenAtom(0, "H"), HiddenAtom(1, "O")),
        show_unit_cell=2,
    )

    projection = projection_module.project_periodic_display(
        atoms,
        context.periodic_display,
        context.config,
        context.hidden_atom_indices,
    )

    assert projection.natoms == 0
    assert projection.positions_2d.shape == (0, 2)
    assert projection.image_shifts.shape == (0, 3)
    assert projection.cell_vertices_2d.shape == (8, 2)
    assert np.isfinite([projection.width, projection.height, projection.scale]).all()
    assert projection.width > 0
    assert projection.height > 0


def test_default_projection_keeps_positive_z_toward_screen_top():
    """默认 2D 视角应与 Plotly 水平观察时的 z 轴向上一致。"""
    atoms = Atoms("HO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])

    projection = project_atoms(atoms, RenderConfig(show_unit_cell=0))

    assert projection.positions_2d[1, 1] > projection.positions_2d[0, 1]

def test_projected_cell_vertices_have_eight_2d_points():
    """显示完整晶胞时，对外返回值必须是 8 个二维顶点。"""
    atoms = Atoms(
        "H",
        positions=[[0.5, 0.5, 0.5]],
        cell=[2.0, 3.0, 4.0],
        pbc=True,
    )

    projection = project_atoms(atoms, RenderConfig(show_unit_cell=2))

    assert projection.cell_vertices_2d.shape == (8, 2)


def test_render_config_atom_radii_validate_copy_and_empty_input():
    source = {"H": 0.4}
    configured = RenderConfig(resolved_element_radii_angstrom=source)
    with pytest.raises(TypeError):
        configured.resolved_element_radii_angstrom["O"] = 0.9
    assert configured.get_atom_radii([]).shape == (0,)
    assert configured.get_atom_radii(["H", "H"]) == pytest.approx([0.4, 0.4])
    with pytest.raises(ValueError, match="resolved.*invalid.*X"):
        RenderConfig(resolved_element_radii_angstrom={"X": 0.4})
    with pytest.raises(ValueError, match="resolved.*missing.*O"):
        configured.get_atom_radii(["O"])


def test_render_config_atom_radii_use_resolved_mapping_or_legacy_default():
    assert RenderConfig().get_atom_radii(["H", "O"]) == pytest.approx(
        covalent_radii[[1, 8]] * 0.6
    )

    configured = RenderConfig(
        resolved_element_radii_angstrom={"H": 0.4, "O": 0.9}
    )

    assert configured.get_atom_radii(["O", "H", "O"]) == pytest.approx(
        [0.9, 0.4, 0.9]
    )
    assert configured.get_atom_radii([]).shape == (0,)
    source = {"H": 0.4}
    frozen = RenderConfig(resolved_element_radii_angstrom=source)
    source["H"] = 9.0
    assert frozen.get_atom_radii(["H"]) == pytest.approx([0.4])
    with pytest.raises(ValueError, match="resolved.*missing.*Si"):
        frozen.get_atom_radii(["Si"])
    with pytest.raises(ValueError, match="resolved.*invalid.*X"):
        RenderConfig(resolved_element_radii_angstrom={"X": 0.4})
    with pytest.raises(ValueError, match="resolved.*H.*0"):
        RenderConfig(resolved_element_radii_angstrom={"H": 0})


def test_projection_matches_ase(sample_atoms):
    """验证投影坐标与 ASE plot_atoms 完全一致。"""
    atoms = sample_atoms
    config = RenderConfig(rotation="0x,0y,0z", show_unit_cell=2)

    fig_ase, ax_ase = plt.subplots(figsize=(8, 6), dpi=150)
    custom_radii = [covalent_radii[a.number] * 0.6 for a in atoms]
    plot_atoms(atoms, ax_ase, radii=custom_radii, rotation="0x,0y,0z")

    ase_circles = [
        (p.get_center()[0], p.get_center()[1], p.get_radius())
        for p in ax_ase.patches if isinstance(p, MplCircle)
    ]
    ase_sorted = sorted(ase_circles)

    proj = project_atoms(atoms, config)
    our_sorted = sorted([
        (proj.positions_2d[i][0], proj.positions_2d[i][1], proj.radii_2d[i])
        for i in range(proj.natoms)
    ])

    assert len(ase_sorted) == len(our_sorted)

    max_err = 0
    for i in range(len(ase_sorted)):
        for d in range(3):
            err = abs(ase_sorted[i][d] - our_sorted[i][d])
            max_err = max(max_err, err)

    print(f"\n[投影对比] {len(ase_sorted)} 个原子，最大误差: {max_err:.10f}")
    assert max_err < 1e-6
    plt.close(fig_ase)


def test_bond_detection(sample_atoms):
    """验证化学键识别（含元素对过滤）。"""
    atoms = sample_atoms
    config = RenderConfig(bond_cutoff=1.0)

    allowed = config.get_effective_allowed_pairs()
    bonds = find_bonds(atoms, config, allowed_pairs=allowed)

    print(f"\n[化学键识别]")
    print(f"  原子数: {len(atoms)}")
    print(f"  允许元素对: {sorted(allowed)}")
    print(f"  识别到 {len(bonds)} 根键")

    symbols = atoms.get_chemical_symbols()
    bond_types = {}
    for b in bonds:
        key = "-".join(sorted([symbols[b.i], symbols[b.j]]))
        bond_types[key] = bond_types.get(key, 0) + 1

    print("  键类型统计:")
    for bt, count in sorted(bond_types.items()):
        print(f"    {bt}: {count}")

    assert len(bonds) > 0
    # Ca-Ca 不应在允许列表中
    assert bond_types.get("Ca-Ca", 0) == 0, "Ca-Ca bonds should be filtered out"


def test_full_render(sample_atoms, tmp_path):
    """完整渲染并导出 SVG。"""
    atoms = sample_atoms
    config = RenderConfig(rotation="90x", bond_cutoff=1.0)

    settings = initialize_bond_settings(atoms, config)
    resolution = resolve_bonds(atoms, settings)
    display = build_periodic_display(
        atoms,
        resolution.matched,
        CellPeriodicSettings(show_unit_cell=config.show_unit_cell),
    )
    proj = projection_module.project_periodic_display(
        atoms,
        display,
        config,
        frozenset(),
    )
    bond_geoms = compute_bond_geometries(
        display.bond_instances,
        proj,
        config,
    )
    fig = render(proj, bond_geoms, config)

    svg_bytes = export_svg(fig, config)
    assert svg_bytes is not None and len(svg_bytes) > 0

    out_path = tmp_path / "meia_test_output.svg"
    with out_path.open("wb") as f:
        f.write(svg_bytes)

    mean_r = proj.radii_2d.mean()
    bond_w = mean_r * config.bond_width_ratio

    print(f"\n[完整渲染]")
    print(f"  旋转: {config.rotation}")
    print(f"  原子数: {proj.natoms}")
    print(f"  键数: {len(resolution.visible)}")
    print(f"  平均原子半径: {mean_r:.4f}")
    print(f"  键宽: {bond_w:.4f} (ratio={config.bond_width_ratio})")
    print(f"  画布: {proj.width:.2f} x {proj.height:.2f}")
    print(f"  SVG 大小: {len(svg_bytes)} bytes")
    print(f"  输出路径: {out_path}")
    plt.close(fig)


def test_render_without_bonds(sample_atoms, tmp_path):
    """仅原子渲染（无键），与 ASE 对比。"""
    atoms = sample_atoms
    config = RenderConfig(rotation="90x", show_unit_cell=0)

    proj = project_atoms(atoms, config)
    fig = render(proj, [], config)

    out_path = tmp_path / "meia_no_bonds.svg"
    export_svg(fig, config, filepath=out_path, overwrite=True)

    print(f"\n[仅原子渲染] 输出路径: {out_path}")
    plt.close(fig)
