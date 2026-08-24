"""O–H···O 氢键识别与跨视图渲染回归。"""

from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
import numpy as np
import pytest

from ase import Atoms

from meia import HydrogenBondSettings
from meia.atom_styles import (
    AtomColorStrength,
    AtomHydrogenBondOverride,
    AtomSelectionSettings,
    HiddenAtom,
    apply_color_strength,
)
from meia.bond_rules import BondPairRule, BondSettings, ResolvedBond
from meia.bond_rules import OverrideVisibility
from meia.config import RenderConfig
from meia.export import export_figure
from meia.i18n import I18n, Locale
from meia.periodic_display import (
    CellPeriodicSettings,
    PeriodicRange,
    build_periodic_display,
)
from meia.size_profiles import CovalentSizeProfile, SizeProfileSettings
from meia.view import render_2d
from meia.viewer import create_3d_figure as _create_3d_figure
from meia.visual_state import (
    AtomCellSettings,
    BondModuleSettings,
    PortableStyle,
    VisualizationState,
    resolve_render_context,
)
import meia.hydrogen_bonds as hydrogen_bond_module


def create_3d_figure(*args, **kwargs):
    kwargs.setdefault(
        "figure_messages",
        I18n(Locale.ZH_CN).bundle("figure3d"),
    )
    return _create_3d_figure(*args, **kwargs)


def _oh_bond(
    donor_oxygen: int = 0,
    hydrogen: int = 1,
    *,
    offset: tuple[int, int, int] = (0, 0, 0),
    visible: bool = True,
) -> ResolvedBond:
    return ResolvedBond(
        i=donor_oxygen,
        j=hydrogen,
        offset=offset,
        distance=1.15,
        pair=("H", "O"),
        bond_id="bond_oh",
        visible=visible,
        visibility_source="pair_enabled" if visible else "atom_hide",
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"draw": 1}, "显示开关"),
        ({"max_hydrogen_oxygen_distance": 0.0}, "必须大于 0"),
        ({"min_angle_degrees": 181.0}, "0°–180°"),
    ],
)
def test_hydrogen_bond_settings_validate(kwargs, message):
    """错误的已应用阈值若被接受，氢键显示会悄然偏离物理约束。"""
    with pytest.raises((TypeError, ValueError), match=message):
        HydrogenBondSettings(**kwargs)


def test_hydrogen_thresholds_have_exact_english_diagnostics():
    with pytest.raises(ValueError) as distance_error:
        HydrogenBondSettings(max_hydrogen_oxygen_distance=0.0)
    assert I18n(Locale.EN).error_text(
        distance_error.value, "bonds.apply_failed"
    ) == "The maximum H···O distance must be greater than 0 Å; received 0.0."

    with pytest.raises(ValueError) as angle_error:
        HydrogenBondSettings(min_angle_degrees=181.0)
    assert I18n(Locale.EN).error_text(
        angle_error.value, "bonds.apply_failed"
    ) == (
        "The minimum O–H···O angle must be from 0° to 180°; "
        "received 181.0°."
    )


def test_render_context_uses_applied_hydrogen_distance_and_global_draw():
    """若使用普通键可见性或默认阈值，已应用氢键设置会被忽略。"""
    atoms = Atoms("OHO", positions=[[0, 0, 0], [1, 0, 0], [3.4, 0, 0]])
    base = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                draw_bonds=False,
                pair_rules=(BondPairRule("H", "O", 0.8, 1.2, enabled=False),),
                hydrogen_bonds=HydrogenBondSettings(True, 2.5, 120.0),
            )
        )
    )
    assert len(resolve_render_context(atoms, base).hydrogen_bonds) == 1
    shorter = replace(
        base,
        style=replace(
            base.style,
            bonds=replace(
                base.style.bonds,
                hydrogen_bonds=HydrogenBondSettings(True, 2.0, 120.0),
            ),
        ),
    )
    assert resolve_render_context(atoms, shorter).hydrogen_bonds == ()
    hidden = replace(
        base,
        style=replace(
            base.style,
            bonds=replace(
                base.style.bonds,
                hydrogen_bonds=replace(base.style.bonds.hydrogen_bonds, draw=False),
            ),
        ),
    )
    assert resolve_render_context(atoms, hidden).hydrogen_bonds == ()


def test_periodic_candidate_keeps_signed_offset_and_exact_distance_boundary():
    """丢失受体像偏移或偷偷加入 O···O cutoff 时必须失败。"""
    atoms = Atoms(
        "OHO",
        positions=[[3.15, 0.0, 0.0], [2.0, 0.0, 0.0], [9.5, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, False, False],
    )

    candidates = hydrogen_bond_module.resolve_hydrogen_bond_candidates(
        atoms,
        (_oh_bond(visible=False),),
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.donor_oxygen == 0
    assert candidate.hydrogen == 1
    assert candidate.acceptor_oxygen == 2
    assert candidate.donor_oxygen_offset_from_hydrogen == (0, 0, 0)
    assert candidate.acceptor_offset_from_hydrogen == (-1, 0, 0)
    assert candidate.hydrogen_acceptor_distance == pytest.approx(2.5)
    assert candidate.angle_degrees == pytest.approx(180.0)
    assert 1.15 + candidate.hydrogen_acceptor_distance > 3.5


def test_periodic_candidate_rejects_distance_just_above_2_5_angstrom():
    """把 2.5 Å 上限写成宽松近似比较时必须失败。"""
    atoms = Atoms(
        "OHO",
        positions=[[3.15, 0.0, 0.0], [2.0, 0.0, 0.0], [9.499, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, False, False],
    )

    assert hydrogen_bond_module.resolve_hydrogen_bond_candidates(
        atoms,
        (_oh_bond(),),
    ) == ()


def test_periodic_candidate_rejects_direct_nextafter_above_distance_limit():
    """2.5 Å 的直接浮点后继不得被 cutoff 的 ULP 扩展误收。"""
    just_above = float(np.nextafter(2.5, np.inf))
    atoms = Atoms(
        "OHO",
        positions=[[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [just_above, 0.0, 0.0]],
    )

    assert just_above > 2.5
    assert hydrogen_bond_module.resolve_hydrogen_bond_candidates(
        atoms,
        (_oh_bond(),),
    ) == ()


def test_same_source_acceptor_keeps_distinct_periodic_images():
    """按源索引去重会错误合并同一受体 O 的两个物理周期像。"""
    atoms = Atoms(
        "OHO",
        positions=[[0.2, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.0, 0.0]],
        cell=[1.0, 8.0, 8.0],
        pbc=[True, False, False],
    )
    donor = _oh_bond(offset=(-1, 0, 0))

    candidates = hydrogen_bond_module.resolve_hydrogen_bond_candidates(
        atoms,
        (donor,),
    )
    acceptor_images = {
        candidate.acceptor_offset_from_hydrogen
        for candidate in candidates
        if candidate.acceptor_oxygen == 2
    }

    assert {(-1, 0, 0), (-2, 0, 0)} <= acceptor_images
    ids = {
        candidate.hydrogen_bond_id
        for candidate in candidates
        if candidate.acceptor_oxygen == 2
    }
    assert len(ids) == len(
        [candidate for candidate in candidates if candidate.acceptor_oxygen == 2]
    )


@pytest.mark.parametrize(
    ("angle_degrees", "expected_count"),
    ((120.0, 1), (119.9, 0)),
)
def test_periodic_candidate_keeps_exact_120_degree_angle_boundary(
    angle_degrees,
    expected_count,
):
    """把 120° 临界点排除或放宽到临界点以下时必须失败。"""
    import math

    radians = math.radians(angle_degrees)
    atoms = Atoms(
        "OHO",
        positions=[
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [2.0 * math.cos(radians), 2.0 * math.sin(radians), 0.0],
        ],
    )

    candidates = hydrogen_bond_module.resolve_hydrogen_bond_candidates(
        atoms,
        (_oh_bond(),),
    )

    assert len(candidates) == expected_count
    if candidates:
        assert candidates[0].angle_degrees == pytest.approx(120.0)


def test_candidate_orients_periodic_donor_offset_from_hydrogen():
    """O→H 共价键偏移若未反向为 H→O，角度与实例端点都会错误。"""
    atoms = Atoms(
        "OHO",
        positions=[[9.5, 0.0, 0.0], [0.5, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, False, False],
    )

    candidate = hydrogen_bond_module.resolve_hydrogen_bond_candidates(
        atoms,
        (_oh_bond(offset=(1, 0, 0)),),
    )[0]

    assert candidate.donor_oxygen_offset_from_hydrogen == (-1, 0, 0)
    assert candidate.acceptor_offset_from_hydrogen == (0, 0, 0)
    assert candidate.angle_degrees == pytest.approx(180.0)


def _periodic_hydrogen_bond_fixture():
    atoms = Atoms(
        "OHO",
        positions=[[3.15, 0.0, 0.0], [2.0, 0.0, 0.0], [9.5, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, False, False],
    )
    matched = (_oh_bond(),)
    candidates = hydrogen_bond_module.resolve_hydrogen_bond_candidates(
        atoms,
        matched,
    )
    display = build_periodic_display(
        atoms,
        matched,
        CellPeriodicSettings(
            show_unit_cell=0,
            a=PeriodicRange(-1, 1),
        ),
    )
    return atoms, matched, candidates, display


def test_periodic_instance_applies_show_and_minimum_participant_strength():
    """只读取受体例外或只读取端点强度时必须失败。"""
    atoms, _, candidates, display = _periodic_hydrogen_bond_fixture()
    selection = AtomSelectionSettings(
        color_strengths=(AtomColorStrength(1, "H", 0.30),),
        hydrogen_bond_overrides=(
            AtomHydrogenBondOverride(2, "O", OverrideVisibility.SHOW),
        ),
    )

    instances = hydrogen_bond_module.instantiate_periodic_hydrogen_bonds(
        atoms,
        display,
        candidates,
        selection,
        {1: 0.30},
    )

    assert len(instances) == 1
    instance = instances[0]
    assert instance.donor_oxygen_key == (0, (0, 0, 0))
    assert instance.hydrogen_key == (1, (0, 0, 0))
    assert instance.acceptor_oxygen_key == (2, (-1, 0, 0))
    assert instance.visible is True
    assert instance.visibility_source == "atom_show"
    assert instance.color_strength == pytest.approx(0.30)
    assert instance.color == apply_color_strength(
        hydrogen_bond_module.HYDROGEN_BOND_COLOR,
        0.30,
    )


def test_periodic_instance_any_hide_wins_over_any_show():
    """三个参与者中任一 HIDE 未压过另一参与者 SHOW 时必须失败。"""
    atoms, _, candidates, display = _periodic_hydrogen_bond_fixture()
    selection = AtomSelectionSettings(
        hydrogen_bond_overrides=(
            AtomHydrogenBondOverride(0, "O", OverrideVisibility.SHOW),
            AtomHydrogenBondOverride(1, "H", OverrideVisibility.HIDE),
        ),
    )

    instances = hydrogen_bond_module.instantiate_periodic_hydrogen_bonds(
        atoms,
        display,
        candidates,
        selection,
        {},
    )

    assert len(instances) == 1
    assert instances[0].visible is False
    assert instances[0].visibility_source == "atom_hide"


def test_periodic_instance_removes_hidden_participant_and_missing_replica():
    """隐藏参与者或范围外受体仍生成虚线时必须失败。"""
    atoms, matched, candidates, display = _periodic_hydrogen_bond_fixture()
    hidden = AtomSelectionSettings(hidden_atoms=(HiddenAtom(2, "O"),))

    assert hydrogen_bond_module.instantiate_periodic_hydrogen_bonds(
        atoms,
        display,
        candidates,
        hidden,
        {},
    ) == ()

    primary_only = build_periodic_display(
        atoms,
        matched,
        CellPeriodicSettings(show_unit_cell=0),
    )
    forced_show = AtomSelectionSettings(
        hydrogen_bond_overrides=(
            AtomHydrogenBondOverride(2, "O", OverrideVisibility.SHOW),
        ),
    )
    assert hydrogen_bond_module.instantiate_periodic_hydrogen_bonds(
        atoms,
        primary_only,
        candidates,
        forced_show,
        {},
    ) == ()


def test_force_show_cannot_create_a_geometry_ineligible_candidate():
    """强制 SHOW 若绕过候选硬门槛，空候选断言必须失败。"""
    atoms = Atoms(
        "OHO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.501, 0.0, 0.0]],
    )
    matched = (_oh_bond(),)
    candidates = hydrogen_bond_module.resolve_hydrogen_bond_candidates(
        atoms,
        matched,
    )
    display = build_periodic_display(
        atoms,
        matched,
        CellPeriodicSettings(show_unit_cell=0),
    )
    selection = AtomSelectionSettings(
        hydrogen_bond_overrides=(
            AtomHydrogenBondOverride(2, "O", OverrideVisibility.SHOW),
        ),
    )

    assert candidates == ()
    assert hydrogen_bond_module.instantiate_periodic_hydrogen_bonds(
        atoms,
        display,
        candidates,
        selection,
        {},
    ) == ()


def test_render_context_is_the_single_periodic_hydrogen_bond_source_for_2d():
    """2D 若重算主单胞氢键，会丢失实例 key、强度与跨边界线段。"""
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
    hydrogen_lines = [
        line
        for line in figure.axes[0].lines
        if isinstance(line, Line2D)
        and str(line.get_gid()).startswith("hydrogen_bond_")
    ]

    assert len(context.hydrogen_bonds) == 1
    assert len(hydrogen_lines) == 1
    assert hydrogen_lines[0].get_gid() == context.hydrogen_bonds[0].instance_id
    assert hydrogen_lines[0].get_color() == context.hydrogen_bonds[0].color

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_same_hydrogen_bond_is_dashed_in_2d_and_3d():
    """2D 与 3D 必须共用同一根 H···O 虚线结果。"""
    atoms = Atoms(
        "OHO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    )
    settings = BondSettings(
        pair_rules=(BondPairRule("H", "O", 0.8, 1.2, enabled=True),),
    )
    config = RenderConfig(show_unit_cell=0)

    figure_2d = render_2d(atoms, config, bond_settings=settings)
    figure_3d = create_3d_figure(atoms, config, bond_settings=settings)

    hydrogen_lines = [
        line
        for line in figure_2d.axes[0].lines
        if isinstance(line, Line2D)
        and str(line.get_gid()).startswith("hydrogen_bond_")
    ]
    hydrogen_traces = [
        trace
        for trace in figure_3d.data
        if trace.meta and trace.meta.get("meia_role") == "hydrogen_bonds"
    ]
    assert len(hydrogen_lines) == 1
    assert hydrogen_lines[0].get_linestyle() == "--"
    assert len(hydrogen_traces) == 1
    assert hydrogen_traces[0].line.dash == "dash"
    assert list(hydrogen_traces[0].customdata)[0][:3] == [0, 1, 2]
    import matplotlib.pyplot as plt
    plt.close(figure_2d)


def test_3d_hydrogen_bonds_use_periodic_instances_and_final_color_groups():
    """重算主单胞氢键或把不同强度压成固定颜色时必须失败。"""
    atoms = Atoms(
        "OHOOHO",
        positions=[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
        ],
        cell=[20.0, 8.0, 8.0],
        pbc=[True, False, False],
    )
    state = VisualizationState(
        style=PortableStyle(
            size_profiles=SizeProfileSettings(
                covalent=CovalentSizeProfile(global_scale=0.5)
            ),
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.2),),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=0,
                a=PeriodicRange(0, 2),
            ),
        ),
        atom_selection=AtomSelectionSettings(
            color_strengths=(
                AtomColorStrength(1, "H", 0.30),
                AtomColorStrength(4, "H", 0.70),
            ),
        ),
    )
    context = resolve_render_context(atoms, state)

    figure = create_3d_figure(
        atoms,
        context.config,
        render_context=context,
    )
    traces = [
        trace for trace in figure.data
        if trace.meta and trace.meta.get("meia_role") == "hydrogen_bonds"
    ]

    expected_colors = {item.color for item in context.hydrogen_bonds}
    assert len(context.hydrogen_bonds) == 4
    assert len(expected_colors) == 2
    assert {trace.line.color for trace in traces} == expected_colors
    assert all(trace.line.dash == "dash" for trace in traces)
    assert all(
        trace.meta["meia_base_line_width"]
        == hydrogen_bond_module.HYDROGEN_BOND_3D_WIDTH
        for trace in traces
    )
    assert {
        tuple(float(value) for value in trace.x[index:index + 2])
        for trace in traces
        for index in range(0, len(trace.x), 3)
    } == {
        (1.155, 2.67),
        (11.155, 12.67),
        (21.155, 22.67),
        (31.155, 32.67),
    }
    identities = [
        trace.customdata[index][3]
        for trace in traces
        for index in range(0, len(trace.customdata), 3)
    ]
    assert len(identities) == len(set(identities)) == 4


def test_svg_export_preserves_hydrogen_bond_identity_and_dash_pattern():
    """SVG 导出不得把 2D 氢键虚线压成无语义实线。"""
    atoms = Atoms(
        "OHO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
    )
    settings = BondSettings(
        pair_rules=(BondPairRule("H", "O", 0.8, 1.2, enabled=True),),
    )
    config = RenderConfig(show_unit_cell=0)

    figure = render_2d(atoms, config, bond_settings=settings)
    svg = export_figure(figure, "svg", config)

    assert b"hydrogen_bond_" in svg
    assert b"stroke-dasharray" in svg

    import matplotlib.pyplot as plt
    plt.close(figure)
