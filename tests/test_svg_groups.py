"""Illustrator 可编辑 SVG 的单键编组契约测试。"""

from dataclasses import replace
from xml.etree import ElementTree as ET

from ase import Atoms

from meia.atom_styles import AtomColorStrength, AtomSelectionSettings
from meia.bond_rules import BondPairRule, BondSettings
from meia.config import RenderConfig
from meia.export import export_figure
from meia.i18n import I18n, Locale
from meia.periodic_display import CellPeriodicSettings, PeriodicRange
import meia.projection as projection_module
from meia.view import render_2d
import meia.view as view_module
import meia.viewer as viewer_module
from meia.viewer import create_3d_figure as _create_3d_figure
from meia.visual_state import (
    BondModuleSettings,
    PortableStyle,
    VisualizationState,
    resolve_render_context,
)


def create_3d_figure(*args, **kwargs):
    kwargs.setdefault(
        "figure_messages",
        I18n(Locale.ZH_CN).bundle("figure3d"),
    )
    return _create_3d_figure(*args, **kwargs)


def _bond_groups(svg_bytes: bytes):
    root = ET.fromstring(svg_bytes)
    return [
        node
        for node in root.iter()
        if node.attrib.get("id", "").startswith("bond_")
        and "data-atom-a" in node.attrib
    ]


def _hydrogen_bond_groups(svg_bytes: bytes):
    root = ET.fromstring(svg_bytes)
    return [
        node
        for node in root.iter()
        if "data-donor-oxygen" in node.attrib
        and "data-hydrogen" in node.attrib
        and "data-acceptor-oxygen" in node.attrib
    ]


def test_2d_3d_and_svg_share_non_topology_bond_instance_ids(monkeypatch):
    """任一视图重新匹配键或重建周期图都应立即失败。"""
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
    expected_ids = {
        instance.bond_instance_id
        for instance in context.periodic_display.bond_instances
        if instance.source_bond.visible
    }

    def fail_recomputation(*_args, **_kwargs):
        raise AssertionError("渲染器不得重新解析化学键、周期拓扑或氢键候选")

    monkeypatch.setattr(view_module, "resolve_bonds", fail_recomputation)
    monkeypatch.setattr(view_module, "build_periodic_display", fail_recomputation)
    monkeypatch.setattr(
        view_module,
        "resolve_hydrogen_bond_candidates",
        fail_recomputation,
    )
    monkeypatch.setattr(
        view_module,
        "instantiate_periodic_hydrogen_bonds",
        fail_recomputation,
    )
    monkeypatch.setattr(viewer_module, "resolve_bonds", fail_recomputation)
    monkeypatch.setattr(viewer_module, "build_periodic_display", fail_recomputation)
    monkeypatch.setattr(viewer_module, "resolve_hydrogen_bonds", fail_recomputation)
    monkeypatch.setattr(
        viewer_module,
        "instantiate_periodic_hydrogen_bonds",
        fail_recomputation,
    )

    figure_3d = create_3d_figure(
        atoms,
        context.config,
        render_context=context,
    )
    figure_2d = render_2d(atoms, context.config, render_context=context)
    svg_groups = _bond_groups(export_figure(figure_2d, "svg", context.config))
    plotly_ids = {
        row[0]
        for trace in figure_3d.data
        if trace.meta and trace.meta.get("meia_role") == "bonds"
        for row in trace.customdata
        if row is not None
    }

    assert not context.periodic_display.diagnostics
    assert {bond.pair for bond in context.periodic_topology_bonds} == {("O", "Si")}
    assert any(
        instance.source_bond.pair == ("Ca", "O")
        for instance in context.periodic_display.bond_instances
    )
    assert any(trace.name.startswith("化学键") for trace in figure_3d.data)
    assert expected_ids
    assert set(figure_2d._meia_bond_manifest) == expected_ids
    assert plotly_ids == expected_ids
    assert {group.attrib["id"] for group in svg_groups} == expected_ids

    import matplotlib.pyplot as plt

    plt.close(figure_2d)


def test_periodic_svg_annotates_each_existing_atom_wrapper_once():
    """SVG 后处理若复制圆或丢失显示像元数据，实例集合断言必须失败。"""
    atoms = Atoms(
        "HHO",
        scaled_positions=[[0.01, 0.5, 0.5], [0.92, 0.5, 0.5], [0.95, 0.5, 0.5]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    state = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.0, 1.2),),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=0,
                a=PeriodicRange(-1, 2),
            ),
        )
    )
    context = resolve_render_context(atoms, state)
    projection = projection_module.project_periodic_display(
        atoms,
        context.periodic_display,
        context.config,
        context.hidden_atom_indices,
    )

    figure = render_2d(atoms, context.config, render_context=context)
    svg = export_figure(figure, "svg", context.config)
    root = ET.fromstring(svg)
    atom_nodes = root.findall(".//*[@data-meia-source-atom-index]")

    assert len(atom_nodes) == projection.natoms
    assert len({node.attrib["id"] for node in atom_nodes}) == projection.natoms
    assert {
        (
            int(node.attrib["data-meia-source-atom-index"]),
            tuple(
                int(value)
                for value in node.attrib["data-meia-image-shift"].split(",")
            ),
        )
        for node in atom_nodes
    } == {
        (int(source), tuple(int(value) for value in image_shift))
        for source, image_shift in zip(
            projection.source_atom_indices,
            projection.image_shifts,
        )
    }
    assert all(len(list(node)) == 1 for node in atom_nodes)

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_periodic_svg_hydrogen_bond_keeps_participant_image_identity():
    """SVG 氢键若只保留源索引而丢失三个显示像身份必须失败。"""
    atoms = Atoms(
        "OHO",
        positions=[[3.15, 0.0, 0.0], [2.0, 0.0, 0.0], [9.5, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, False, False],
    )
    state = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.2),),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=0,
                a=PeriodicRange(-1, 1),
            ),
        ),
        atom_selection=AtomSelectionSettings(
            color_strengths=(AtomColorStrength(1, "H", 0.30),),
        ),
    )
    context = resolve_render_context(atoms, state)

    figure = render_2d(atoms, context.config, render_context=context)
    groups = _hydrogen_bond_groups(
        export_figure(figure, "svg", context.config)
    )

    assert len(groups) == 1
    group = groups[0]
    assert group.attrib["id"] == context.hydrogen_bonds[0].instance_id
    assert group.attrib["data-donor-oxygen"] == "0"
    assert group.attrib["data-hydrogen"] == "1"
    assert group.attrib["data-acceptor-oxygen"] == "2"
    assert group.attrib["data-donor-oxygen-image-shift"] == "0,0,0"
    assert group.attrib["data-hydrogen-image-shift"] == "0,0,0"
    assert group.attrib["data-acceptor-oxygen-image-shift"] == "-1,0,0"
    assert len(list(group)) == 1

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_multi_replica_hydrogen_bond_ids_match_svg_manifest_one_to_one():
    """多副本氢键若复用 ID，SVG manifest 会丢失物理实例。"""
    atoms = Atoms(
        "OHO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[10.0, 8.0, 8.0],
        pbc=[True, False, False],
    )
    state = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.2),),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=0,
                a=PeriodicRange(0, 3),
            ),
        ),
    )
    context = resolve_render_context(atoms, state)

    figure = render_2d(atoms, context.config, render_context=context)
    groups = _hydrogen_bond_groups(export_figure(figure, "svg", context.config))
    expected_ids = [item.instance_id for item in context.hydrogen_bonds]
    manifest_ids = list(figure._meia_hydrogen_bond_manifest)
    svg_ids = [group.attrib["id"] for group in groups]

    assert len(expected_ids) == 3
    assert len(set(expected_ids)) == 3
    assert manifest_ids == expected_ids
    assert svg_ids == expected_ids

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_svg_has_one_named_group_with_six_visible_children_per_bond():
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    settings = BondSettings(
        pair_rules=(BondPairRule("C", "O", 1.0, 1.4, enabled=True),),
    )
    config = RenderConfig(show_unit_cell=0)
    fig = render_2d(atoms, config, bond_settings=settings)

    groups = _bond_groups(export_figure(fig, "svg", config))

    assert len(groups) == 1
    group = groups[0]
    assert group.attrib["data-atom-a"] == "0"
    assert group.attrib["data-atom-b"] == "1"
    assert group.attrib["data-elements"] == "C-O"
    assert group.attrib["data-periodic-offset"] == "0,0,0"
    assert [child.attrib["data-role"] for child in group] == [
        "cap-a",
        "cap-b",
        "rect-a",
        "rect-b",
        "outline-1",
        "outline-2",
    ]


def test_svg_bond_group_ids_are_unique():
    atoms = Atoms(
        "COC",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [2.4, 0.0, 0.0]],
    )
    settings = BondSettings(
        pair_rules=(BondPairRule("C", "O", 1.0, 1.4, enabled=True),),
    )
    config = RenderConfig(show_unit_cell=0)
    fig = render_2d(atoms, config, bond_settings=settings)

    groups = _bond_groups(export_figure(fig, "svg", config))
    group_ids = [group.attrib["id"] for group in groups]

    assert len(groups) == 2
    assert len(group_ids) == len(set(group_ids))
    assert all(len(group) == 6 for group in groups)


def test_svg_renders_unwrapped_periodic_bond_with_endpoint_image_identity():
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [9.6, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    settings = BondSettings(
        pair_rules=(BondPairRule("H", "H", 0.4, 0.6, enabled=True),),
    )
    config = RenderConfig(rotation="0x,0y,0z", show_unit_cell=0)
    fig = render_2d(atoms, config, bond_settings=settings)

    groups = _bond_groups(export_figure(fig, "svg", config))

    assert len(groups) == 1
    assert len(groups[0]) == 6
    assert groups[0].attrib["data-periodic-offset"] == "-1,0,0"
    assert groups[0].attrib["data-atom-a-image-shift"] == "0,0,0"
    assert groups[0].attrib["data-atom-b-image-shift"] == "-1,0,0"


def test_direct_export_refuses_to_overwrite_without_explicit_opt_in(tmp_path):
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    config = RenderConfig(show_unit_cell=0)
    fig = render_2d(atoms, config)
    target = tmp_path / "existing.svg"
    target.write_text("keep-me", encoding="utf-8")

    try:
        import pytest

        with pytest.raises(FileExistsError, match="existing\\.svg"):
            export_figure(fig, "svg", config, str(target))
        assert target.read_text(encoding="utf-8") == "keep-me"

        export_figure(fig, "svg", config, str(target), overwrite=True)
        assert target.read_bytes().startswith(b"<?xml")
    finally:
        import matplotlib.pyplot as plt

        plt.close(fig)
