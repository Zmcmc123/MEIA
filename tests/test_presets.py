"""严格 v7 通用风格与工作状态快照 JSON 测试。"""

import ast
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.data import chemical_symbols
from ase.data.colors import jmol_colors
from matplotlib.colors import to_hex

from meia.atom_styles import (
    AtomColorOverride,
    AtomColorStrength,
    AtomHydrogenBondOverride,
    AtomSelectionSettings,
    HiddenAtom,
    compact_color_strengths,
    resolved_color_strengths,
)
from meia.bond_rules import (
    AtomBondOverride,
    BondPairRule,
    BondStrokeStyle,
    OverrideVisibility,
)
from meia.hydrogen_bonds import HydrogenBondSettings
from meia.i18n import I18n, Locale
from meia.periodic_display import CellPeriodicSettings, PeriodicRange
from meia.size_profiles import (
    CovalentSizeProfile,
    RadiusMode as ProfileRadiusMode,
    SizeProfileSettings,
    UniformSizeProfile,
)
from meia.presets import (
    SCHEMA_VERSION,
    PresetError,
    PresetKind,
    PresetMetadata,
    SnapshotStructure,
    StylePreset,
    WorkspaceSnapshot,
    apply_style_preset,
    load_default_style,
    parse_preset,
    style_preset_to_json,
    workspace_snapshot_to_json,
    visual_state_fingerprint,
)
from meia.view_state import CameraState
from meia.visual_state import (
    AtomCellSettings,
    BondModuleSettings,
    ExportSettings,
    PairRuleDefaults,
    PortableStyle,
    ViewSettings,
    VisualizationState,
)
from meia.workspace import build_style_preset, build_workspace_snapshot


def test_every_runtime_preset_error_declares_a_translation_key():
    project_root = Path(__file__).resolve().parents[1]
    missing_keys = []
    for path in (project_root / "app.py", *sorted((project_root / "meia").rglob("*.py"))):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "PresetError"
            ):
                continue
            if not any(keyword.arg == "message_key" for keyword in node.keywords):
                missing_keys.append(f"{path.relative_to(project_root)}:{node.lineno}")

    assert missing_keys == []


def test_v7_round_trip_stores_both_size_profiles_without_duplicate_size_fields():
    profiles = SizeProfileSettings(
        active_mode=ProfileRadiusMode.UNIFORM,
        covalent=CovalentSizeProfile(
            global_scale=0.72,
            reference_overrides_angstrom={"O": 0.8},
            bond_width_ratio=0.31,
        ),
        uniform=UniformSizeProfile(
            global_scale=1.1,
            reference_radius_angstrom=0.30,
            reference_overrides_angstrom={"H": 0.25},
            bond_width_ratio=0.57,
        ),
    )
    style = replace(load_default_style().style, size_profiles=profiles)
    preset = build_style_preset(VisualizationState(style), "dual", "0.11.0")

    payload = style_preset_to_json(preset)
    decoded = json.loads(payload)
    recovered = parse_preset(payload)

    assert decoded["schema_version"] == 7
    assert decoded["size_profiles"] == {
        "active_mode": "uniform",
        "covalent": {
            "global_scale": 0.72,
            "reference_overrides_angstrom": {"O": 0.8},
            "bond_width_ratio": 0.31,
        },
        "uniform": {
            "global_scale": 1.1,
            "reference_radius_angstrom": 0.30,
            "reference_overrides_angstrom": {"H": 0.25},
            "bond_width_ratio": 0.57,
        },
    }
    assert set(decoded["atoms"]) == {"outline_width", "element_colors"}
    assert set(decoded["bonds"]["style"]) == {"stroke_width", "stroke_color"}
    assert recovered.style.size_profiles == profiles


def test_v7_style_and_workspace_round_trip_periodic_and_atom_visibility():
    style = replace(
        load_default_style().style,
        cell_periodic=CellPeriodicSettings(
            show_unit_cell=1,
            unwrap_bonded_groups=False,
            a=PeriodicRange(-1, 2),
        ),
    )
    selection = AtomSelectionSettings(
        hidden_atoms=(HiddenAtom(0, "H"),),
        hydrogen_bond_overrides=(
            AtomHydrogenBondOverride(1, "O", OverrideVisibility.SHOW),
        ),
    )
    state = VisualizationState(style=style, atom_selection=selection)
    assert parse_preset(
        style_preset_to_json(build_style_preset(state, "style", "0.11.0"))
    ).style == style
    snapshot = build_workspace_snapshot(
        Atoms("HO", positions=[[0, 0, 0], [1, 0, 0]]),
        "fixture.xyz",
        state,
        "workspace",
        "0.11.0",
    )
    assert parse_preset(workspace_snapshot_to_json(snapshot)).state == state


def test_v7_round_trip_preserves_topology_defaults_and_hydrogen_settings():
    style = replace(
        load_default_style().style,
        bonds=BondModuleSettings(
            defaults=PairRuleDefaults(long_distance_threshold_angstrom=2.0),
            hydrogen_bonds=HydrogenBondSettings(False, 2.3, 135.0),
            pair_rules=(
                BondPairRule(
                    "Ca",
                    "O",
                    0.0,
                    2.42,
                    enabled=True,
                    participates_in_periodic_unwrap=False,
                ),
            ),
        ),
    )
    preset = StylePreset(
        PresetMetadata(
            SCHEMA_VERSION,
            PresetKind.STYLE,
            "v7",
            "2026-08-23T00:00:00+08:00",
            "0.11.0",
        ),
        style,
    )
    decoded = json.loads(style_preset_to_json(preset))
    assert decoded["bonds"]["pair_rule_defaults"][
        "long_distance_threshold_angstrom"
    ] == 2.0
    assert decoded["bonds"]["hydrogen_bonds"] == {
        "draw": False,
        "max_hydrogen_oxygen_distance_angstrom": 2.3,
        "min_angle_degrees": 135.0,
    }
    assert decoded["bonds"]["pair_rules"][0][
        "participates_in_periodic_unwrap"
    ] is False
    assert parse_preset(style_preset_to_json(preset)).style == style


@pytest.mark.parametrize("version", [1, 2, 3, 4, 5, 6, 8])
def test_only_schema_v7_is_accepted(version):
    payload = json.loads(style_preset_to_json(load_default_style()))
    payload["schema_version"] = version
    with pytest.raises(PresetError, match="仅支持 v7"):
        parse_preset(json.dumps(payload))


def test_negative_outline_width_has_stable_exact_english_diagnostic():
    payload = json.loads(style_preset_to_json(load_default_style()))
    payload["atoms"]["outline_width"] = -1

    with pytest.raises(PresetError) as captured:
        parse_preset(json.dumps(payload))

    assert captured.value.message_key == "atom.outline_nonnegative"
    assert I18n(Locale.EN).error_text(
        captured.value,
        "file.style_invalid",
    ) == "Atom outline width must be 0 or greater; received -1.0."


def test_invalid_created_at_and_incomplete_palette_keep_exact_reason():
    invalid_time = json.loads(style_preset_to_json(load_default_style()))
    invalid_time["created_at"] = "not-a-time"
    with pytest.raises(PresetError) as time_error:
        parse_preset(json.dumps(invalid_time))
    assert I18n(Locale.EN).error_text(
        time_error.value, "file.style_invalid"
    ) == "Preset field created_at must be an ISO 8601 timestamp; received 'not-a-time'."

    incomplete_palette = json.loads(style_preset_to_json(load_default_style()))
    incomplete_palette["atoms"]["element_colors"].pop("H")
    with pytest.raises(PresetError) as palette_error:
        parse_preset(json.dumps(incomplete_palette))
    assert I18n(Locale.EN).error_text(
        palette_error.value, "file.style_invalid"
    ) == "The element palette is incomplete; missing: H; invalid extras: none."


@pytest.mark.parametrize(
    ("section", "field", "value", "expected"),
    [
        (
            "view",
            "rotation",
            "not-a-rotation",
            "Preset field view.rotation must be a valid ASE rotation string; "
            "received 'not-a-rotation'.",
        ),
        (
            "export",
            "format",
            "jpeg",
            "Preset field export.format must be one of svg, png, pdf; "
            "received 'jpeg'.",
        ),
        (
            "bonds",
            "pair_rule_defaults.bond_cutoff",
            0,
            "Preset field bonds.pair_rule_defaults.bond_cutoff must be greater "
            "than 0; received 0.",
        ),
    ],
)
def test_known_style_field_errors_keep_field_value_and_reason(
    section,
    field,
    value,
    expected,
):
    payload = json.loads(style_preset_to_json(load_default_style()))
    target = payload[section]
    parts = field.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value

    with pytest.raises(PresetError) as error:
        parse_preset(json.dumps(payload))

    assert I18n(Locale.EN).error_text(error.value, "file.style_invalid") == expected


def test_v7_round_trip_preserves_both_size_profiles():
    profiles = SizeProfileSettings(
        active_mode=ProfileRadiusMode.UNIFORM,
        covalent=CovalentSizeProfile(
            global_scale=0.75,
            reference_overrides_angstrom={"H": 0.4, "O": 0.8},
            bond_width_ratio=0.31,
        ),
        uniform=UniformSizeProfile(
            global_scale=0.9,
            reference_radius_angstrom=1.1,
            reference_overrides_angstrom={"O": 1.3, "Si": 1.6},
            bond_width_ratio=0.57,
        ),
    )
    style = replace(
        load_default_style().style,
        size_profiles=profiles,
    )
    preset = build_style_preset(VisualizationState(style), "radius", "0.11.0")

    recovered = parse_preset(style_preset_to_json(preset))

    assert recovered.style.size_profiles == profiles


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda profiles: profiles.pop("uniform"), "缺少字段"),
        (lambda profiles: profiles.update(radius_scale=0.6), "未知字段"),
        (lambda profiles: profiles["uniform"].update(typo=1), "未知字段"),
        (lambda profiles: profiles.pop("active_mode"), "缺少字段"),
        (lambda profiles: profiles.update(active_mode="vdw"), "半径模式"),
        (lambda profiles: profiles["uniform"].update(global_scale=True), "数值"),
        (lambda profiles: profiles["uniform"].update(global_scale=0), "大于 0"),
        (
            lambda profiles: profiles["uniform"].update(
                reference_radius_angstrom=np.inf
            ),
            "有限",
        ),
        (
            lambda profiles: profiles["covalent"].update(
                reference_overrides_angstrom=[]
            ),
            "JSON 对象",
        ),
        (
            lambda profiles: profiles["uniform"].update(
                reference_overrides_angstrom={"NotAnElement": 1.0}
            ),
            "非法元素",
        ),
        (
            lambda profiles: profiles["uniform"].update(
                reference_overrides_angstrom={"O": -1.0}
            ),
            "大于 0",
        ),
    ],
)
def test_v7_size_profiles_strictly_reject_malformed_content(mutator, message):
    data = json.loads(style_preset_to_json(_style_preset()))
    mutator(data["size_profiles"])

    with pytest.raises(PresetError, match=message):
        parse_preset(json.dumps(data, allow_nan=True))


def test_v7_size_profiles_reject_duplicate_nested_keys():
    payload = style_preset_to_json(_style_preset())
    duplicate_scalar = payload.replace(
        '"global_scale": 0.65,',
        '"global_scale": 0.65,\n      "global_scale": 0.65,',
        1,
    )
    duplicate_override = payload.replace(
        '"H": 0.4,',
        '"H": 0.4,\n        "H": 0.4,',
        1,
    )

    with pytest.raises(PresetError, match="重复字段"):
        parse_preset(duplicate_scalar)
    with pytest.raises(PresetError, match="重复字段"):
        parse_preset(duplicate_override)


def _complete_test_palette() -> dict[str, str]:
    return {
        symbol: to_hex(
            jmol_colors[index if index < len(jmol_colors) else index - 32]
        ).upper()
        for index, symbol in enumerate(chemical_symbols[1:119], start=1)
    }


def _portable_style() -> PortableStyle:
    return PortableStyle(
        view=ViewSettings(
            rotation="-90x",
            camera=CameraState(eye=(0.0, 2.0, 0.0)),
        ),
        size_profiles=SizeProfileSettings(
            active_mode=ProfileRadiusMode.UNIFORM,
            covalent=CovalentSizeProfile(
                global_scale=0.65,
                reference_overrides_angstrom={"H": 0.4, "O": 0.8},
                bond_width_ratio=0.35,
            ),
            uniform=UniformSizeProfile(
                global_scale=0.65,
                reference_radius_angstrom=1.1,
                reference_overrides_angstrom={"O": 1.3, "Si": 1.6},
                bond_width_ratio=0.35,
            ),
        ),
        atom_cell=AtomCellSettings(
            outline_width=0.4,
            element_colors=_complete_test_palette(),
        ),
        bonds=BondModuleSettings(
            draw_bonds=True,
            pair_rules=(BondPairRule("C", "O", 1.0, 1.5),),
            style=BondStrokeStyle(0.15, "#231815"),
        ),
        cell_periodic=CellPeriodicSettings(
            show_unit_cell=2,
            unwrap_bonded_groups=False,
            a=PeriodicRange(-1, 2),
            b=PeriodicRange(0, 2),
        ),
        export=ExportSettings("svg", 600, True),
    )


def _metadata(kind: PresetKind) -> PresetMetadata:
    return PresetMetadata(
        schema_version=SCHEMA_VERSION,
        preset_kind=kind,
        name="paper-style",
        created_at="2026-08-21T18:00:00+08:00",
        meia_version="0.11.0",
    )


def _style_preset() -> StylePreset:
    return StylePreset(_metadata(PresetKind.STYLE), _portable_style())


def _workspace_snapshot() -> WorkspaceSnapshot:
    atoms = Atoms(
        "COO",
        positions=[[0, 0, 0], [1.2, 0, 0], [2.4, 0, 0]],
        cell=[[8, 0, 0], [0, 9, 0], [0, 0, 10]],
        pbc=[True, True, False],
    )
    selection = AtomSelectionSettings(
        selected_atom_indices=(1, 2),
        color_overrides=(AtomColorOverride(1, "O", "#336699"),),
        color_strengths=(AtomColorStrength(2, "O", 0.3),),
        bond_overrides=(
            AtomBondOverride(1, "O", "C", "O", OverrideVisibility.HIDE),
        ),
        hidden_atoms=(HiddenAtom(0, "C"),),
        hydrogen_bond_overrides=(
            AtomHydrogenBondOverride(2, "O", OverrideVisibility.SHOW),
        ),
    )
    return WorkspaceSnapshot(
        metadata=_metadata(PresetKind.WORKSPACE_SNAPSHOT),
        structure=SnapshotStructure.from_atoms(atoms, "CONTCAR"),
        state=VisualizationState(_portable_style(), selection),
    )


def test_style_import_normalizes_non_periodic_axes_before_returning_candidate():
    atoms = Atoms("CO", cell=[8, 9, 10], pbc=[True, False, False])
    current = VisualizationState(load_default_style().style)
    imported = replace(
        _style_preset(),
        style=replace(
            _style_preset().style,
            cell_periodic=CellPeriodicSettings(
                a=PeriodicRange(-1, 2),
                b=PeriodicRange(-50_000, 50_000),
                c=PeriodicRange(7, 9),
            ),
        ),
    )

    candidate = apply_style_preset(current, imported, atoms)

    assert candidate.style.cell_periodic.a == PeriodicRange(-1, 2)
    assert candidate.style.cell_periodic.b == PeriodicRange(0, 1)
    assert candidate.style.cell_periodic.c == PeriodicRange(0, 1)


def test_style_import_rejects_normalized_display_over_instance_limit():
    atoms = Atoms("CO", cell=[8, 9, 10], pbc=[True, False, False])
    current = VisualizationState(load_default_style().style)
    imported = replace(
        _style_preset(),
        style=replace(
            _style_preset().style,
            cell_periodic=CellPeriodicSettings(
                a=PeriodicRange(0, 25_001),
                b=PeriodicRange(-50_000, 50_000),
            ),
        ),
    )

    with pytest.raises(PresetError, match="50,000"):
        apply_style_preset(current, imported, atoms)


def test_style_json_has_exact_periodic_fragment_and_no_atom_index_fields():
    preset = _style_preset()

    payload = style_preset_to_json(preset)
    decoded = json.loads(payload)

    assert decoded["schema_version"] == 7
    assert decoded["preset_kind"] == "style"
    assert decoded["size_profiles"] == {
        "active_mode": "uniform",
        "covalent": {
            "global_scale": 0.65,
            "reference_overrides_angstrom": {"H": 0.4, "O": 0.8},
            "bond_width_ratio": 0.35,
        },
        "uniform": {
            "global_scale": 0.65,
            "reference_radius_angstrom": 1.1,
            "reference_overrides_angstrom": {"O": 1.3, "Si": 1.6},
            "bond_width_ratio": 0.35,
        },
    }
    assert decoded["cell_periodic"] == {
        "show_unit_cell": 2,
        "unwrap_bonded_groups": False,
        "ranges": {
            "a": {"start": -1, "end": 2},
            "b": {"start": 0, "end": 2},
            "c": {"start": 0, "end": 1},
        },
    }
    assert "structure" not in decoded
    assert "atom_selection" not in decoded
    assert "atom_index" not in payload
    assert parse_preset(payload) == preset


def test_workspace_snapshot_round_trip_restores_structure_and_all_atom_state():
    snapshot = _workspace_snapshot()

    payload = workspace_snapshot_to_json(snapshot)
    decoded = json.loads(payload)
    recovered = parse_preset(payload)

    assert decoded["atom_selection"]["hidden_atoms"] == [
        {"atom_index": 0, "atom_symbol": "C"}
    ]
    assert decoded["atom_selection"]["hydrogen_bond_overrides"] == [
        {"atom_index": 2, "atom_symbol": "O", "visibility": "show"}
    ]
    assert isinstance(recovered, WorkspaceSnapshot)
    atoms = recovered.structure.to_atoms()
    assert atoms.get_chemical_symbols() == ["C", "O", "O"]
    assert np.array_equal(atoms.positions, [[0, 0, 0], [1.2, 0, 0], [2.4, 0, 0]])
    assert np.array_equal(atoms.pbc, [True, True, False])
    assert recovered == snapshot


def test_v7_workspace_expands_compact_background_and_recompacts_on_import():
    atoms = Atoms(symbols=["H"] * 5000, positions=np.zeros((5000, 3)))
    selection = AtomSelectionSettings(
        selected_atom_indices=(3, 7),
        color_strengths=(
            AtomColorStrength(3, "H", 1.0),
            AtomColorStrength(7, "H", 1.0),
        ),
        default_color_strength=0.30,
    )
    snapshot = WorkspaceSnapshot(
        metadata=_metadata(PresetKind.WORKSPACE_SNAPSHOT),
        structure=SnapshotStructure.from_atoms(atoms, "large.xyz"),
        state=VisualizationState(load_default_style().style, selection),
    )

    payload = workspace_snapshot_to_json(snapshot)
    decoded = json.loads(payload)
    recovered = parse_preset(payload)

    assert set(decoded["atom_selection"]) == {
        "selected_indices",
        "color_overrides",
        "color_strengths",
        "bond_overrides",
        "hidden_atoms",
        "hydrogen_bond_overrides",
    }
    assert decoded["schema_version"] == 7
    assert len(decoded["atom_selection"]["color_strengths"]) == 4998
    assert isinstance(recovered, WorkspaceSnapshot)
    recovered_settings = recovered.state.atom_selection
    assert recovered_settings.default_color_strength == pytest.approx(0.30)
    assert [
        (item.atom_index, item.strength)
        for item in recovered_settings.color_strengths
    ] == [(3, 1.0), (7, 1.0)]
    assert np.array_equal(
        resolved_color_strengths(recovered_settings, len(atoms)),
        resolved_color_strengths(selection, len(atoms)),
    )


def test_normalized_equivalent_strength_profiles_share_state_fingerprint():
    symbols = ("H", "H", "H", "H")
    expanded = AtomSelectionSettings(
        color_strengths=(
            AtomColorStrength(0, "H", 0.30),
            AtomColorStrength(1, "H", 0.30),
            AtomColorStrength(2, "H", 0.30),
        )
    )
    default, exceptions = compact_color_strengths(
        symbols,
        resolved_color_strengths(expanded, len(symbols)),
    )
    normalized = AtomSelectionSettings(
        color_strengths=exceptions,
        default_color_strength=default,
    )
    compact = AtomSelectionSettings(
        color_strengths=(AtomColorStrength(3, "H", 1.0),),
        default_color_strength=0.30,
    )

    assert normalized == compact
    assert visual_state_fingerprint(
        VisualizationState(atom_selection=normalized)
    ) == visual_state_fingerprint(VisualizationState(atom_selection=compact))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda data: data.update(unknown=True), "未知字段"),
        (lambda data: data.pop("cell_periodic"), "缺少字段"),
        (lambda data: data["cell_periodic"]["ranges"].pop("a"), "缺少字段"),
        (
            lambda data: data["cell_periodic"]["ranges"]["a"].update(end=-1),
            "终点",
        ),
        (
            lambda data: data["structure"]["positions_angstrom"][0].__setitem__(
                0, np.nan
            ),
            "有限",
        ),
        (lambda data: data["structure"].update(pbc=[True, 1, False]), "PBC"),
        (
            lambda data: data["atom_selection"]["color_strengths"][0].update(
                atom_index=99
            ),
            "范围",
        ),
        (
            lambda data: data["atom_selection"]["hidden_atoms"][0].update(
                atom_symbol="H"
            ),
            "元素",
        ),
        (
            lambda data: data["atom_selection"]["hydrogen_bond_overrides"][
                0
            ].update(visibility="maybe"),
            "氢键",
        ),
        (
            lambda data: data["atom_selection"]["hidden_atoms"][0].update(
                typo=True
            ),
            "未知字段",
        ),
    ],
)
def test_workspace_snapshot_strictly_rejects_malformed_content(mutator, message):
    data = json.loads(workspace_snapshot_to_json(_workspace_snapshot()))
    mutator(data)

    with pytest.raises(PresetError, match=message):
        parse_preset(json.dumps(data, allow_nan=True))


@pytest.mark.parametrize(
    "camera",
    [
        {},
        {
            "eye": {"x": 1, "y": 1, "z": 1},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
            "projection": {"type": "orthographic"},
            "typo": True,
        },
        {
            "eye": {"x": 1, "y": 1},
            "up": {"x": 0, "y": 0, "z": 1},
            "center": {"x": 0, "y": 0, "z": 0},
            "projection": {"type": "orthographic"},
        },
    ],
)
def test_preset_camera_requires_exact_complete_nested_fields(camera):
    data = json.loads(style_preset_to_json(_style_preset()))
    data["view"]["camera"] = camera

    with pytest.raises(PresetError, match="camera"):
        parse_preset(json.dumps(data))


@pytest.mark.parametrize(
    "mutator",
    [
        lambda bonds: bonds.pop("hydrogen_bonds"),
        lambda bonds: bonds["pair_rule_defaults"].pop(
            "long_distance_threshold_angstrom"
        ),
        lambda bonds: bonds["pair_rules"][0].pop(
            "participates_in_periodic_unwrap"
        ),
    ],
)
def test_v7_bond_sections_require_all_topology_and_hydrogen_fields(mutator):
    data = json.loads(style_preset_to_json(_style_preset()))
    mutator(data["bonds"])

    with pytest.raises(PresetError, match="缺少字段"):
        parse_preset(json.dumps(data))


@pytest.mark.parametrize("invalid_coordinate", [True, "1.0"])
def test_preset_camera_rejects_boolean_and_numeric_string_coordinates(
    invalid_coordinate,
):
    data = json.loads(style_preset_to_json(_style_preset()))
    data["view"]["camera"]["eye"]["x"] = invalid_coordinate

    with pytest.raises(PresetError, match=r"view\.camera\.eye\.x"):
        parse_preset(json.dumps(data))


def test_invalid_and_duplicate_json_are_rejected():
    with pytest.raises(PresetError, match="JSON"):
        parse_preset("{not-json")

    payload = style_preset_to_json(_style_preset())
    duplicate = payload.replace(
        '"schema_version": 7,',
        '"schema_version": 7,\n  "schema_version": 7,',
        1,
    )
    with pytest.raises(PresetError, match="重复字段"):
        parse_preset(duplicate)
