"""内置完整周期表风格与确定性生成器测试。"""

import hashlib
from pathlib import Path
import re
import subprocess
import sys
from xml.etree import ElementTree as ET

import pytest
from ase.data import chemical_symbols

import meia
from meia.export import export_figure
from meia.hydrogen_bonds import HydrogenBondSettings
from meia.presets import (
    SCHEMA_VERSION,
    StylePreset,
    WorkspaceSnapshot,
    load_default_style,
    parse_preset,
    style_preset_to_json,
)
from meia.visual_state import resolve_render_context
from meia.size_profiles import SizeProfileSettings
from meia.view import render_2d


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STYLE_PATH = (
    PROJECT_ROOT / "meia" / "defaults" / "default_style.meia.json"
)


def _without_svg_date(payload: bytes) -> bytes:
    return re.sub(
        br"<dc:date>.*?</dc:date>",
        b"<dc:date></dc:date>",
        payload,
    )


def test_default_style_contains_exactly_all_real_elements():
    preset = load_default_style()

    assert set(preset.style.atom_cell.element_colors) == set(
        chemical_symbols[1:119]
    )


def test_default_style_keeps_confirmed_anchor_colours():
    colors = load_default_style().style.atom_cell.element_colors

    assert {key: colors[key] for key in ["H", "C", "O", "Si", "Ca"]} == {
        "H": "#E6E6E5",
        "C": "#3F4F6A",
        "O": "#E5A6A6",
        "Si": "#5386C6",
        "Ca": "#9ECC91",
    }


def test_default_style_supplies_every_adjustable_global_initial_value():
    preset = load_default_style()
    style = preset.style

    assert style.view.rotation == "-90x"
    assert style.size_profiles == SizeProfileSettings()
    assert style.atom_cell.outline_width == 0.5
    assert style.cell_periodic.show_unit_cell == 2
    assert style.cell_periodic.unwrap_bonded_groups is True
    assert style.cell_periodic.a.start == 0
    assert style.cell_periodic.a.end == 1
    assert style.bonds.draw_bonds is True
    assert style.size_profiles.covalent.bond_width_ratio == 0.45
    assert style.size_profiles.uniform.bond_width_ratio == 0.45
    assert style.bonds.style.stroke_width == 0.25
    assert style.bonds.style.stroke_color == "#231815"
    assert style.bonds.defaults.bond_cutoff == 1.0
    assert style.bonds.defaults.pair_distance_multipliers == (("H", "O", 1.2),)
    assert style.export.format == "svg"
    assert style.export.dpi == 600
    assert style.export.transparent is True


def test_default_style_uses_public_parser_and_is_stable_under_round_trip():
    parsed_file = parse_preset(DEFAULT_STYLE_PATH.read_bytes())
    loaded = load_default_style()

    assert isinstance(parsed_file, StylePreset)
    assert parsed_file.metadata.schema_version == SCHEMA_VERSION
    assert parsed_file == loaded
    assert parse_preset(style_preset_to_json(loaded)) == loaded


def test_default_v7_style_is_generic_and_preserves_h_o_multiplier():
    preset = load_default_style()

    assert preset.metadata.schema_version == 7
    assert preset.metadata.meia_version == "0.11.0"
    assert preset.style.bonds.pair_rules == ()
    assert preset.style.bonds.defaults.long_distance_threshold_angstrom == 2.0
    assert preset.style.bonds.defaults.multiplier_mapping()[("H", "O")] == 1.2
    assert preset.style.bonds.hydrogen_bonds == HydrogenBondSettings()


@pytest.mark.release
def test_committed_reference_case_is_current_v7_workspace():
    source_path = PROJECT_ROOT / "examples" / "CONTCAR"
    workspace_path = (
        PROJECT_ROOT / "examples" / "meia-visual-state.workspace.meia.json"
    )
    output_path = PROJECT_ROOT / "examples" / "CONTCAR_meia-2.svg"
    parsed = parse_preset(workspace_path.read_bytes())

    assert isinstance(parsed, WorkspaceSnapshot)
    assert parsed.metadata.schema_version == 7
    assert parsed.metadata.meia_version == "0.11.0"
    assert parsed.metadata.created_at == "2026-08-24T20:01:07+08:00"
    assert parsed.structure.source_name == "CONTCAR"
    assert len(parsed.structure.symbols) == 225
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == (
        "187ee6a6d1c5bffc2b55a8ea254f0dc86c82a1f56743fcdad55504488b399d5f"
    )
    assert len(parsed.state.atom_selection.color_strengths) == 51
    assert parsed.state.atom_selection.selected_atom_indices == (46, 47)
    assert len(parsed.state.atom_selection.bond_overrides) == 2
    assert parsed.state.style.size_profiles == SizeProfileSettings()
    assert parsed.state.style.view.rotation == "-90x"
    assert parsed.state.style.bonds.hydrogen_bonds == HydrogenBondSettings(
        draw=True,
        max_hydrogen_oxygen_distance=2.5,
        min_angle_degrees=120.0,
    )

    by_pair = {rule.pair: rule for rule in parsed.state.style.bonds.pair_rules}
    assert set(by_pair) == {
        ("C", "O"),
        ("Ca", "Ca"),
        ("Ca", "O"),
        ("H", "O"),
        ("O", "Si"),
    }
    for pair in (("Ca", "Ca"), ("Ca", "O")):
        assert by_pair[pair].enabled is False
        assert by_pair[pair].participates_in_periodic_unwrap is False
    for pair in (("C", "O"), ("H", "O"), ("O", "Si")):
        assert by_pair[pair].enabled is True
        assert by_pair[pair].participates_in_periodic_unwrap is True

    atoms = parsed.structure.to_atoms()
    context = resolve_render_context(atoms, parsed.state)
    assert not context.periodic_display.diagnostics
    assert dict(context.bond_resolution.match_counts) == {
        ("C", "O"): 2,
        ("Ca", "Ca"): 23,
        ("Ca", "O"): 275,
        ("H", "O"): 35,
        ("O", "Si"): 96,
    }
    assert len(context.bond_resolution.visible) == 138
    assert len(context.hydrogen_bonds) == 21

    rendered = render_2d(atoms, context.config, render_context=context)
    regenerated_svg = export_figure(rendered, "svg", context.config)
    rendered.clear()
    assert _without_svg_date(regenerated_svg) == _without_svg_date(
        output_path.read_bytes()
    )

    source_oxygen_indices = {117, 121, 127, 128, 140, 141}
    connections = {
        index: [
            bond
            for bond in context.periodic_topology_bonds
            if bond.pair == ("O", "Si") and index in (bond.i, bond.j)
        ]
        for index in source_oxygen_indices
    }
    assert all(connections.values())
    for bonds in connections.values():
        assert all(
            tuple(
                right - left
                for left, right in zip(
                    context.periodic_display.base_image_shifts[bond.i],
                    context.periodic_display.base_image_shifts[bond.j],
                )
            )
            == bond.offset
            for bond in bonds
        )

    svg_root = ET.fromstring(output_path.read_bytes())
    svg_namespace = "{http://www.w3.org/2000/svg}"
    assert svg_root.tag == f"{svg_namespace}svg"
    axes = next(
        node for node in svg_root.iter(f"{svg_namespace}g")
        if node.attrib.get("id") == "axes_1"
    )
    atom_groups = [
        node for node in axes
        if node.attrib.get("id", "").startswith("atom_")
    ]
    bond_groups = [
        node for node in axes
        if node.attrib.get("id", "").startswith("bond_")
    ]
    cell_groups = [
        node for node in axes
        if node.attrib.get("id", "").startswith("patch_")
    ]
    assert len(atom_groups) == 225
    assert {
        int(node.attrib["data-meia-source-atom-index"])
        for node in atom_groups
    } == set(range(225))
    assert all(len(node) == 1 for node in atom_groups)
    assert len(bond_groups) == 138
    assert {node.attrib["data-elements"] for node in bond_groups} == {
        "C-O",
        "Ca-O",
        "H-O",
        "O-Si",
    }
    assert all(len(node) == 6 for node in bond_groups)
    assert len(cell_groups) == 738
    assert all(
        len(node) == 1
        and "stroke-dasharray" in node[0].attrib.get("style", "")
        and "stroke: #808080" in node[0].attrib.get("style", "")
        for node in cell_groups
    )


@pytest.mark.release
def test_default_style_and_generator_report_meia_0_11_0(tmp_path):
    """包版本或生成元数据未升级到 0.11.0 时必须失败。"""
    assert meia.__version__ == "0.11.0"
    assert load_default_style().metadata.meia_version == "0.11.0"

    output = tmp_path / "generated.meia.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "generate_default_style.py"),
            "--output",
            str(output),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    generated = parse_preset(output.read_bytes())
    assert isinstance(generated, StylePreset)
    assert generated.metadata.meia_version == "0.11.0"
    assert generated.metadata.schema_version == 7
    assert generated.style.size_profiles == SizeProfileSettings()
    assert parse_preset(style_preset_to_json(generated)) == generated
    assert output.read_bytes() == DEFAULT_STYLE_PATH.read_bytes()


@pytest.mark.release
def test_palette_generator_refuses_silent_overwrite_and_is_deterministic(tmp_path):
    script = PROJECT_ROOT / "scripts" / "generate_default_style.py"
    output = tmp_path / "generated.meia.json"
    command = [sys.executable, str(script), "--output", str(output)]

    first = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    first_bytes = output.read_bytes()

    refused = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert refused.returncode != 0
    assert "--overwrite" in (refused.stderr + refused.stdout)
    assert output.read_bytes() == first_bytes

    replaced = subprocess.run(
        command + ["--overwrite"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert replaced.returncode == 0, replaced.stderr
    assert output.read_bytes() == first_bytes
