"""
Streamlit AppTest 与批量处理脚本测试。
"""

import os
import re
import sys
import shutil
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from xml.etree import ElementTree as ET
import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse, Polygon
from streamlit.testing.v1 import AppTest

from ase import Atoms
from ase.data import chemical_symbols
from ase.utils import rotate

import app as app_module
import meia.projection as projection_module
import meia.visual_state as visual_state_module
import scripts.regenerate_visualization_example as regeneration_script
from meia.atom_styles import (
    AtomColorStrength,
    AtomSelectionSettings,
    HiddenAtom,
)
from meia.config import RenderConfig
from meia.bond_rules import (
    BondPairRule,
    BondResolution,
    BondSettings,
    BondStrokeStyle,
    BondStyle,
    ResolvedBond,
)
from meia.batch import (
    _style_preset_with_overrides,
    batch_process,
    find_structure_files,
    main as batch_main,
)
from meia.export import export_figure
from meia.geometry import compute_bond_geometries
from meia.hydrogen_bonds import HydrogenBondSettings
from meia.i18n import I18n, Locale
from meia.locale_state import (
    APP_LOCALE_KEY,
    APP_LOCALE_SOURCE_KEY,
    APP_LOCALE_WIDGET_KEY,
)
from meia.presets import (
    PresetError,
    load_default_style,
    parse_preset,
    style_preset_to_json,
    workspace_snapshot_to_json,
)
from meia.sidebar import AtomFormSubmission, VISUAL_STATE_KEY
from meia.size_profiles import (
    CovalentSizeProfile,
    RadiusMode as ProfileRadiusMode,
    SizeProfileSettings,
    UniformSizeProfile,
)
from meia.view import camera_to_rotation_matrix, render_2d
from meia.periodic_display import (
    CellPeriodicSettings,
    PeriodicDisplayDiagnostic,
    PeriodicRange,
)
from meia.view_state import AppliedViewState, CameraState
from meia.viewer import create_3d_figure as _create_3d_figure
from meia.projection import project_atoms
from meia.visual_state import (
    AtomCellSettings,
    ExportSettings,
    BondModuleSettings,
    ExportSettings as AppliedExportSettings,
    PortableStyle,
    VisualizationState,
    resolve_render_context,
)
from meia.workspace import (
    ActiveWorkspace,
    build_style_preset,
    build_workspace_snapshot,
    stage_snapshot,
)
from app import (
    ACTIVE_WORKSPACE_KEY,
    ATOM_SELECTION_DRAFT_REVISION_KEY,
    HANDLED_SNAPSHOT_HASH_KEY,
    PENDING_SNAPSHOT_KEY,
    PENDING_SNAPSHOT_HASH_KEY,
    RESET_STYLE_BASELINE_KEY,
    RESET_WIDGET_REINITIALIZE_KEY,
    THREE_D_INTERACTION_CAPTION,
    SNAPSHOT_CONFIRMATION_KEY,
    SNAPSHOT_CONFIRMATION_RESET_KEY,
    VISUAL_STRUCTURE_ID_KEY,
    _commit_confirmed_snapshot,
    _consume_reset_widget_reinitialize,
    _consume_snapshot_confirmation_reset,
    _periodic_diagnostic_notice,
    _render_json_imports,
    _render_export_downloads,
    _render_global_forms,
    _clear_reset_scoped_widgets,
    _reset_snapshot_confirmation_for_payload,
    atom_selection_draft_widget_key,
)


def create_3d_figure(*args, **kwargs):
    kwargs.setdefault(
        "figure_messages",
        I18n(Locale.ZH_CN).bundle("figure3d"),
    )
    return _create_3d_figure(*args, **kwargs)


def _app_test(app_path: str, *, locale: Locale = Locale.ZH_CN) -> AppTest:
    app = AppTest.from_file(app_path)
    app.session_state[APP_LOCALE_KEY] = locale.value
    app.session_state[APP_LOCALE_SOURCE_KEY] = "manual"
    return app


def _all_visible_text(app: AppTest) -> str:
    values = []
    for collection_name in (
        "title",
        "markdown",
        "subheader",
        "caption",
        "info",
        "success",
        "warning",
        "error",
    ):
        values.extend(str(item.value) for item in getattr(app, collection_name))
    values.extend(expander.label for expander in app.expander)
    for collection_name in (
        "button",
        "checkbox",
        "file_uploader",
        "multiselect",
        "number_input",
        "radio",
        "selectbox",
        "slider",
        "text_input",
    ):
        collection = (
            getattr(app, collection_name)
            if hasattr(app, collection_name)
            else app.get(collection_name)
        )
        values.extend(item.label for item in collection)
    return "\n".join(values)


def _resolved_fixture_bond(i, j, offset, symbol_i, symbol_j):
    offset_text = "_".join(str(value) for value in offset)
    return ResolvedBond(
        i=i,
        j=j,
        offset=offset,
        distance=1.0,
        pair=tuple(sorted((symbol_i, symbol_j))),
        bond_id=f"bond_{i}_{j}_{offset_text}",
        visible=True,
        visibility_source="pair_enabled",
    )


def _portable_style_with_complete_palette(**changes):
    """为严格预设边界构造与默认 JSON 无关的完整调色板。"""
    atom_cell = AtomCellSettings(
        element_colors={symbol: "#808080" for symbol in chemical_symbols[1:119]}
    )
    return replace(PortableStyle(atom_cell=atom_cell), **changes)


def cross_layer_periodic_fixture():
    symbols = ["C"] * 7 + ["H", "H", "O", "O", "H", "O"]
    scaled = [
        [0.20, 0.20, 0.20],
        [0.40, 0.20, 0.20],
        [0.30, 0.40, 0.20],
        [0.20, 0.20, 0.95],
        [0.40, 0.20, 0.95],
        [0.30, 0.40, 0.95],
        [0.25, 0.25, 0.95],
        [0.01, 0.50, 0.50],
        [0.92, 0.50, 0.50],
        [0.95, 0.50, 0.50],
        [0.50, 0.80, 0.50],
        [0.50, 0.90, 0.50],
        [0.50, 0.15, 0.50],
    ]
    atoms = Atoms(symbols, scaled_positions=scaled, cell=[10, 10, 10], pbc=True)
    matched = (
        _resolved_fixture_bond(0, 1, (0, 0, 0), "C", "C"),
        _resolved_fixture_bond(1, 2, (0, 0, 0), "C", "C"),
        _resolved_fixture_bond(0, 2, (0, 0, 1), "C", "C"),
        _resolved_fixture_bond(0, 3, (0, 0, -1), "C", "C"),
        _resolved_fixture_bond(1, 4, (0, 0, -1), "C", "C"),
        _resolved_fixture_bond(2, 5, (0, 0, -1), "C", "C"),
        _resolved_fixture_bond(0, 6, (0, 0, -1), "C", "C"),
        _resolved_fixture_bond(7, 9, (-1, 0, 0), "H", "O"),
        _resolved_fixture_bond(8, 9, (0, 0, 0), "H", "O"),
        _resolved_fixture_bond(10, 11, (0, 0, 0), "O", "H"),
    )
    return atoms, BondResolution(
        matched=matched,
        visible=matched,
        match_counts={("C", "C"): 7, ("H", "O"): 3},
    )


def _assert_atoms_unchanged(atoms, snapshot):
    positions, cell, pbc, symbols = snapshot
    assert np.array_equal(atoms.positions, positions)
    assert np.array_equal(atoms.cell.array, cell)
    assert np.array_equal(atoms.pbc, pbc)
    assert atoms.get_chemical_symbols() == symbols


def test_first_run_waits_for_invisible_locale_preference():
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    app = AppTest.from_file(app_path).run(timeout=30)

    assert not app.exception
    assert [caption.value for caption in app.caption] == ["MEIA"]
    assert not app.expander


def test_language_switch_preserves_visual_workspace_and_drafts():
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]]),
        "locale-fixture.xyz",
        b"locale-fixture",
    )
    visual_state = VisualizationState(
        load_default_style().style,
        atom_selection=AtomSelectionSettings(selected_atom_indices=(0,)),
    )
    app = _app_test(app_path)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.session_state[VISUAL_STRUCTURE_ID_KEY] = active.structure_id
    app.session_state[VISUAL_STATE_KEY] = visual_state
    app.session_state["meia_export_form_format"] = "PNG"
    app.session_state["meia_preset_name"] = "draft-name"
    app.session_state["meia_cell_periodic_a_start"] = -2
    app.session_state["meia_atom_selection_range"] = "2"
    app.run(timeout=30)

    assert not app.exception
    assert len(app.radio) == 1
    next(
        item for item in app.number_input if item.label == "O 最终显示半径 / Å"
    ).set_value(0.8).run(timeout=30)
    assert next(
        item for item in app.number_input if item.label == "O 最终显示半径 / Å"
    ).value == pytest.approx(0.8)
    app.radio[0].set_value(Locale.EN).run(timeout=30)

    assert not app.exception
    assert app.session_state[APP_LOCALE_KEY] == "en"
    assert app.session_state[APP_LOCALE_SOURCE_KEY] == "manual"
    assert app.session_state[ACTIVE_WORKSPACE_KEY] is active
    assert app.session_state[VISUAL_STATE_KEY] is visual_state
    assert app.session_state[VISUAL_STATE_KEY].atom_selection.selected_atom_indices == (0,)
    assert app.session_state["meia_export_form_format"] == "PNG"
    assert app.session_state["meia_preset_name"] == "draft-name"
    assert app.session_state["meia_cell_periodic_a_start"] == -2
    assert app.session_state["meia_atom_selection_range"] == "2"
    assert next(
        item
        for item in app.number_input
        if item.label == "O Final Display Radius / Å"
    ).value == pytest.approx(0.8)


def test_english_surface_and_meia_title_are_complete():
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]]),
        "english.xyz",
        b"english",
    )
    app = _app_test(app_path, locale=Locale.EN)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.run(timeout=30)

    assert not app.exception
    visible = _all_visible_text(app)
    for label in (
        "📁 File",
        "Atoms",
        "Bonds",
        "Unit Cell & Periodicity",
        "Atom Selection",
        "Export",
        "Interactive 3D Preview",
        "Flattened 2D Output",
        "Show Detected Bonds",
        "Hydrogen Bonds",
    ):
        assert label in visible
    assert "普通化学键" not in visible
    title_markup = [item.value for item in app.markdown if "<h1>" in item.value]
    assert len(title_markup) == 1
    assert "<strong>MEIA</strong>" in title_markup[0]
    assert "- Molecular and Extended-system Illustration Assistant" in title_markup[0]
    assert "font-size:0.55em;font-weight:500" in title_markup[0]
    assert I18n(Locale.EN).text("atom.radius_mode.help") in [
        item.value for item in app.caption
    ]


@pytest.mark.parametrize("locale", [Locale.ZH_CN, Locale.EN])
def test_app_displays_exact_author_credit_in_both_locales(locale):
    """The visible credit must remain identical when the interface language changes."""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    app = _app_test(app_path, locale=locale).run(timeout=30)

    assert not app.exception
    assert [caption.value for caption in app.caption].count(
        "Xiaomei_974 & codex"
    ) == 1


def test_reset_initial_configuration_preserves_manual_english_locale():
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]]),
        "reset-locale.xyz",
        b"reset-locale",
    )
    app = _app_test(app_path, locale=Locale.EN)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.run(timeout=30)

    next(
        button
        for button in app.button
        if button.label == "Restore Initial Settings"
    ).click().run(timeout=30)

    assert not app.exception
    assert app.session_state[APP_LOCALE_KEY] == "en"
    assert app.session_state[APP_LOCALE_SOURCE_KEY] == "manual"


def test_chinese_title_and_radius_help_remain_exact():
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    app = _app_test(app_path).run(timeout=30)

    assert [item.value for item in app.title] == ["⚛ 原子构型可视化"]
    assert I18n(Locale.ZH_CN).text("atom.radius_mode.help") in [
        item.value for item in app.caption
    ]

def test_app_explains_confirmed_camera_sync():
    """页面必须说明按钮确认后才把 3D 视角应用到 2D。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]]),
        "fixture.xyz",
        b"fixture",
    )
    app = _app_test(app_path)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.run(timeout=30)

    captions = [caption.value for caption in app.caption]
    assert THREE_D_INTERACTION_CAPTION in captions
    assert not any("不会自动同步" in caption for caption in captions)


def test_app_exposes_confirmed_bond_defaults_and_two_json_imports():
    """初始界面应区分通用风格和会覆盖结构的工作状态快照。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    app = _app_test(app_path).run(timeout=30)

    sliders = {slider.label: slider.value for slider in app.slider}
    assert sliders["统一键宽比例"] == 0.45
    assert sliders["统一描边粗细"] == 0.25
    assert "成键容差" not in sliders
    uploader_labels = {
        uploader.label for uploader in app.get("file_uploader")
    }
    assert uploader_labels >= {
        "导入通用风格预设（JSON）",
        "导入工作状态快照（JSON）",
    }
    assert "导入完整可视化预设（JSON）" not in uploader_labels
    assert I18n(Locale.ZH_CN).text("file.default_style_path_hint") in [
        caption.value for caption in app.caption
    ]
    apply_buttons = [
        button for button in app.button if button.label == "应用化学键设置"
    ]
    assert len(apply_buttons) == 1
    assert apply_buttons[0].disabled is True


def test_snapshot_commit_preserves_old_session_values_when_validation_rejects():
    old_atoms = Atoms("H", positions=[[0, 0, 0]])
    old_active = ActiveWorkspace.from_upload(old_atoms, "old.xyz", b"old")
    old_state = VisualizationState(load_default_style().style)
    session_state = {
        ACTIVE_WORKSPACE_KEY: old_active,
        "meia_visual_state": old_state,
        VISUAL_STRUCTURE_ID_KEY: old_active.structure_id,
    }
    snapshot_atoms = Atoms(
        "CO",
        positions=[[0, 0, 0], [1.2, 0, 0]],
        cell=[8, 9, 10],
        pbc=[True, False, False],
    )
    invalid_state = VisualizationState(
        replace(
            load_default_style().style,
            cell_periodic=CellPeriodicSettings(a=PeriodicRange(0, 25_001)),
        )
    )
    pending = stage_snapshot(
        build_workspace_snapshot(
            snapshot_atoms,
            "too-large.legacy",
            invalid_state,
            "too-large",
            "0.8.0",
        )
    )

    with pytest.raises(PresetError, match="50,000"):
        _commit_confirmed_snapshot(session_state, pending, confirmed=True)

    assert session_state[ACTIVE_WORKSPACE_KEY] is old_active
    assert session_state["meia_visual_state"] is old_state
    assert session_state[VISUAL_STRUCTURE_ID_KEY] == old_active.structure_id


class _ImportBoundaryStreamlit:
    """只模拟 JSON 导入边界需要的 Streamlit 表单交互。"""

    def __init__(self, session_state, submitted_labels):
        self.session_state = session_state
        self.submitted_labels = set(submitted_labels)
        self.errors = []
        self.rerun_count = 0

    def form(self, *_args, **_kwargs):
        return nullcontext()

    def form_submit_button(self, label, *, disabled=False, **_kwargs):
        return not disabled and label in self.submitted_labels

    def checkbox(self, _label, *, disabled=False, key=None, **_kwargs):
        if disabled:
            return bool(self.session_state.get(key, False))
        self.session_state[key] = True
        return True

    def error(self, message):
        self.errors.append(message)

    def info(self, _message):
        return None

    def warning(self, _message):
        return None

    def rerun(self):
        self.rerun_count += 1


class _JsonUpload:
    def __init__(self, name, payload):
        self.name = name
        self._payload = payload

    def getvalue(self):
        return self._payload


def test_english_json_import_errors_are_localized_and_parameterized(monkeypatch):
    fake = _ImportBoundaryStreamlit({}, set())
    monkeypatch.setattr(app_module, "st", fake)

    _render_json_imports(
        None,
        None,
        _JsonUpload("broken.style.meia.json", b'{"schema_version": 7,'),
        None,
        nullcontext(),
        I18n(Locale.EN),
    )

    assert len(fake.errors) == 1
    assert fake.errors[0].startswith("The preset JSON is invalid:")
    assert "line 1 column" in fake.errors[0]
    assert re.search(r"[\u3400-\u9fff]", fake.errors[0]) is None


def test_english_style_slot_rejects_workspace_without_chinese(monkeypatch):
    state = VisualizationState(_portable_style_with_complete_palette())
    workspace = build_workspace_snapshot(
        Atoms("H", positions=[[0, 0, 0]]),
        "workspace.xyz",
        state,
        "workspace",
        "0.11.0",
    )
    fake = _ImportBoundaryStreamlit({}, set())
    monkeypatch.setattr(app_module, "st", fake)

    _render_json_imports(
        None,
        None,
        _JsonUpload(
            "workspace.workspace.meia.json",
            workspace_snapshot_to_json(workspace).encode("utf-8"),
        ),
        None,
        nullcontext(),
        I18n(Locale.EN),
    )

    assert fake.errors == [
        "This file is a workspace snapshot. Import it through the workspace "
        "snapshot field below."
    ]
    assert re.search(r"[\u3400-\u9fff]", fake.errors[0]) is None


def test_style_reset_baseline_updates_only_after_successful_apply(monkeypatch):
    atoms = Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]])
    active = ActiveWorkspace.from_upload(atoms, "active.xyz", b"active")
    current = VisualizationState(_portable_style_with_complete_palette())
    incoming_style = replace(
        current.style,
        atom_cell=replace(current.style.atom_cell, outline_width=0.85),
    )
    incoming = build_style_preset(
        VisualizationState(incoming_style),
        "incoming",
        "0.11.0",
    )
    old_baseline = PortableStyle(
        atom_cell=replace(current.style.atom_cell, outline_width=0.25)
    )
    session_state = {
        RESET_STYLE_BASELINE_KEY: old_baseline,
        APP_LOCALE_KEY: "en",
        APP_LOCALE_SOURCE_KEY: "manual",
    }
    fake = _ImportBoundaryStreamlit(
        session_state,
        {"应用通用风格预设"},
    )
    monkeypatch.setattr(app_module, "st", fake)
    monkeypatch.setattr(app_module, "parse_preset", lambda _payload: incoming)

    _render_json_imports(
        active,
        current,
        _JsonUpload(
            "incoming-style.meia.json",
            b"incoming-style",
        ),
        None,
        nullcontext(),
    )

    assert session_state[RESET_STYLE_BASELINE_KEY] == incoming_style
    assert session_state["meia_visual_state"].style.atom_cell.outline_width == 0.85
    assert session_state[APP_LOCALE_KEY] == "en"
    assert session_state[APP_LOCALE_SOURCE_KEY] == "manual"
    assert fake.rerun_count == 1


def test_style_upload_without_apply_and_failed_apply_preserve_reset_baseline(
    monkeypatch,
):
    atoms = Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]])
    active = ActiveWorkspace.from_upload(atoms, "active.xyz", b"active")
    current = VisualizationState(_portable_style_with_complete_palette())
    incoming = build_style_preset(current, "incoming", "0.11.0")
    upload = _JsonUpload(
        "incoming-style.meia.json",
        b"incoming-style",
    )
    old_baseline = PortableStyle(
        atom_cell=replace(current.style.atom_cell, outline_width=0.25)
    )

    not_applied_state = {RESET_STYLE_BASELINE_KEY: old_baseline}
    fake = _ImportBoundaryStreamlit(not_applied_state, set())
    monkeypatch.setattr(app_module, "st", fake)
    monkeypatch.setattr(app_module, "parse_preset", lambda _payload: incoming)
    _render_json_imports(active, current, upload, None, nullcontext())
    assert not_applied_state[RESET_STYLE_BASELINE_KEY] is old_baseline
    assert "meia_visual_state" not in not_applied_state

    failed_state = {
        RESET_STYLE_BASELINE_KEY: old_baseline,
        "meia_visual_state": current,
    }
    fake = _ImportBoundaryStreamlit(
        failed_state,
        {"应用通用风格预设"},
    )
    monkeypatch.setattr(app_module, "st", fake)
    monkeypatch.setattr(
        app_module,
        "apply_style_preset",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid style")),
    )
    _render_json_imports(active, current, upload, None, nullcontext())

    assert failed_state[RESET_STYLE_BASELINE_KEY] is old_baseline
    assert failed_state["meia_visual_state"] is current
    assert fake.rerun_count == 0
    assert fake.errors == ["通用风格预设未应用：ValueError: invalid style"]


def test_successful_workspace_snapshot_import_does_not_replace_reset_baseline(
    monkeypatch,
):
    current_atoms = Atoms("H", positions=[[0, 0, 0]])
    active = ActiveWorkspace.from_upload(current_atoms, "active.xyz", b"active")
    current = VisualizationState(_portable_style_with_complete_palette())
    old_baseline = PortableStyle(
        atom_cell=replace(current.style.atom_cell, outline_width=0.25)
    )
    snapshot_state = VisualizationState(
        replace(
            current.style,
            atom_cell=replace(current.style.atom_cell, outline_width=0.95),
        )
    )
    snapshot = build_workspace_snapshot(
        Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]]),
        "snapshot.xyz",
        snapshot_state,
        "snapshot",
        "0.11.0",
    )
    session_state = {
        ACTIVE_WORKSPACE_KEY: active,
        "meia_visual_state": current,
        VISUAL_STRUCTURE_ID_KEY: active.structure_id,
        RESET_STYLE_BASELINE_KEY: old_baseline,
        APP_LOCALE_KEY: "en",
        APP_LOCALE_SOURCE_KEY: "manual",
    }
    fake = _ImportBoundaryStreamlit(
        session_state,
        {"确认导入工作状态快照"},
    )
    monkeypatch.setattr(app_module, "st", fake)
    monkeypatch.setattr(app_module, "parse_preset", lambda _payload: snapshot)

    _render_json_imports(
        active,
        current,
        None,
        _JsonUpload(
            "snapshot.workspace.meia.json",
            b"workspace-snapshot",
        ),
        nullcontext(),
    )

    assert session_state[RESET_STYLE_BASELINE_KEY] is old_baseline
    assert session_state["meia_visual_state"].style.atom_cell.outline_width == 0.95
    assert session_state[APP_LOCALE_KEY] == "en"
    assert session_state[APP_LOCALE_SOURCE_KEY] == "manual"
    assert fake.rerun_count == 1


def test_workspace_uploaded_as_style_is_rejected_before_session_mutation(
    monkeypatch,
):
    """合法工作快照误投风格入口时，预检拒绝不得触碰任何会话状态。"""
    old_active = ActiveWorkspace.from_upload(
        Atoms(
            "H",
            positions=[[0.25, 0.5, 0.75]],
            cell=[4.0, 5.0, 6.0],
            pbc=[True, False, False],
        ),
        "old.xyz",
        b"old-payload",
    )
    old_state = VisualizationState(load_default_style().style)
    existing_pending = stage_snapshot(
        build_workspace_snapshot(
            Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]]),
            "pending.xyz",
            old_state,
            "pending",
            "0.9.0",
        )
    )
    existing_payload = workspace_snapshot_to_json(
        existing_pending.snapshot
    ).encode("utf-8")
    existing_hash = hashlib.sha256(existing_payload).hexdigest()
    session_sentinel = object()
    session_state = {
        ACTIVE_WORKSPACE_KEY: old_active,
        "meia_visual_state": old_state,
        VISUAL_STRUCTURE_ID_KEY: old_active.structure_id,
        PENDING_SNAPSHOT_KEY: existing_pending,
        PENDING_SNAPSHOT_HASH_KEY: existing_hash,
        HANDLED_SNAPSHOT_HASH_KEY: "handled-hash-sentinel",
        SNAPSHOT_CONFIRMATION_KEY: True,
        SNAPSHOT_CONFIRMATION_RESET_KEY: True,
        "unrelated-session-sentinel": session_sentinel,
    }
    before = dict(session_state)
    active_atoms_before = old_active.atoms
    fake = _ImportBoundaryStreamlit(
        session_state,
        {"应用通用风格预设"},
    )
    monkeypatch.setattr(app_module, "st", fake)
    incoming_workspace = build_workspace_snapshot(
        Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]]),
        "incoming.xyz",
        old_state,
        "incoming",
        "0.9.0",
    )

    _render_json_imports(
        old_active,
        old_state,
        _JsonUpload(
            "workspace-in-style-slot.meia.json",
            workspace_snapshot_to_json(incoming_workspace).encode("utf-8"),
        ),
        _JsonUpload("pending.workspace.meia.json", existing_payload),
        nullcontext(),
    )

    assert set(session_state) == set(before)
    for key, original_value in before.items():
        assert session_state[key] is original_value, key
    assert session_state[ACTIVE_WORKSPACE_KEY] is old_active
    assert session_state[ACTIVE_WORKSPACE_KEY].atoms is active_atoms_before
    assert session_state["meia_visual_state"] is old_state
    assert session_state[PENDING_SNAPSHOT_KEY] is existing_pending
    assert session_state["unrelated-session-sentinel"] is session_sentinel
    assert fake.rerun_count == 0
    assert any("该文件是工作状态快照" in message for message in fake.errors)


@pytest.mark.parametrize(
    ("preset_kind", "schema_version"),
    [("style", 6), ("workspace", 6), ("workspace", 8)],
)
def test_rejected_schema_cannot_partially_apply_at_json_import_boundary(
    monkeypatch,
    preset_kind,
    schema_version,
):
    old_atoms = Atoms(
        "H",
        positions=[[0.25, 0.5, 0.75]],
        cell=[4.0, 5.0, 6.0],
        pbc=[True, False, False],
    )
    old_active = ActiveWorkspace.from_upload(old_atoms, "old.xyz", b"old-payload")
    default_style = load_default_style().style
    old_state = VisualizationState(
        replace(
            default_style,
            export=replace(default_style.export, dpi=300),
        )
    )
    existing_pending = stage_snapshot(
        build_workspace_snapshot(
            Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]]),
            "pending.xyz",
            VisualizationState(default_style),
            "pending",
            "0.9.0",
        )
    )
    existing_payload = workspace_snapshot_to_json(
        existing_pending.snapshot
    ).encode("utf-8")
    existing_hash = hashlib.sha256(existing_payload).hexdigest()
    session_state = {
        ACTIVE_WORKSPACE_KEY: old_active,
        "meia_visual_state": old_state,
        VISUAL_STRUCTURE_ID_KEY: old_active.structure_id,
        PENDING_SNAPSHOT_KEY: existing_pending,
        PENDING_SNAPSHOT_HASH_KEY: existing_hash,
        HANDLED_SNAPSHOT_HASH_KEY: "handled-hash-sentinel",
        SNAPSHOT_CONFIRMATION_KEY: True,
        SNAPSHOT_CONFIRMATION_RESET_KEY: True,
    }
    fake = _ImportBoundaryStreamlit(
        session_state,
        {
            {
                "style": "应用通用风格预设",
                "workspace": "确认导入工作状态快照",
            }[preset_kind]
        },
    )
    monkeypatch.setattr(app_module, "st", fake)

    if preset_kind == "style":
        mapping = json.loads(style_preset_to_json(load_default_style()))
    else:
        incoming = build_workspace_snapshot(
            Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]]),
            "incoming.xyz",
            VisualizationState(default_style),
            "incoming",
            "0.9.0",
        )
        mapping = json.loads(workspace_snapshot_to_json(incoming))
    mapping["schema_version"] = schema_version
    upload = _JsonUpload(
        f"rejected-{preset_kind}.meia.json",
        json.dumps(mapping).encode("utf-8"),
    )

    _render_json_imports(
        old_active,
        old_state,
        upload if preset_kind == "style" else None,
        (
            upload
            if preset_kind == "workspace"
            else _JsonUpload("pending.workspace.meia.json", existing_payload)
        ),
        nullcontext(),
    )

    assert session_state[ACTIVE_WORKSPACE_KEY] is old_active
    assert session_state["meia_visual_state"] is old_state
    assert session_state[VISUAL_STRUCTURE_ID_KEY] == old_active.structure_id
    assert session_state[ACTIVE_WORKSPACE_KEY].source_content == b"old-payload"
    assert session_state[ACTIVE_WORKSPACE_KEY].atoms == old_atoms
    assert session_state[PENDING_SNAPSHOT_KEY] is existing_pending
    assert session_state[PENDING_SNAPSHOT_HASH_KEY] == existing_hash
    assert session_state[HANDLED_SNAPSHOT_HASH_KEY] == "handled-hash-sentinel"
    assert session_state[SNAPSHOT_CONFIRMATION_KEY] is True
    assert session_state[SNAPSHOT_CONFIRMATION_RESET_KEY] is True
    assert fake.rerun_count == 0
    assert any("仅支持 v7" in message for message in fake.errors)


def test_periodic_diagnostic_notice_lists_unique_atoms_and_element_pairs():
    context = SimpleNamespace(
        periodic_display=SimpleNamespace(
            diagnostics=(
                PeriodicDisplayDiagnostic(
                    "ambiguous_periodic_attachment",
                    (1, 2, 3),
                    ("b1",),
                    (("O", "Ca"), ("O", "Si")),
                ),
                PeriodicDisplayDiagnostic(
                    "ambiguous_periodic_attachment",
                    (3, 4),
                    ("b2",),
                    (("Ca", "O"),),
                ),
            )
        )
    )

    notice = _periodic_diagnostic_notice(context)

    assert "2 项冲突" in notice
    assert "4 个原子" in notice
    assert "Ca–O、O–Si" in notice
    assert notice.count("Ca–O") == 1
    assert "O–Ca" not in notice
    assert "化学键" in notice
    assert "参与周期整理" in notice
    assert "冲突部分已保守保持原位" in notice


def test_periodic_diagnostic_notice_without_pairs_keeps_fallback_action():
    context = SimpleNamespace(
        periodic_display=SimpleNamespace(
            diagnostics=(
                PeriodicDisplayDiagnostic(
                    "ambiguous_periodic_attachment",
                    (1, 2),
                    ("opaque-id",),
                ),
            )
        )
    )

    notice = _periodic_diagnostic_notice(context)

    assert "1 项冲突" in notice
    assert "2 个原子" in notice
    assert "相关元素对包括" not in notice
    assert "请在“化学键”模块调整" in notice
    assert "冲突部分已保守保持原位" in notice


def test_periodic_diagnostic_notice_is_absent_without_diagnostics():
    context = SimpleNamespace(periodic_display=SimpleNamespace(diagnostics=()))

    assert _periodic_diagnostic_notice(context) is None


@pytest.mark.parametrize(
    ("hydrogen_settings", "expected_hydrogen_count"),
    (
        (HydrogenBondSettings(False, 2.3, 135.0), 0),
        (
            HydrogenBondSettings(True, 2.75, 135.0),
            3,
        ),
    ),
)
def test_app_renderers_keep_one_context_and_fixed_interaction_caption(
    monkeypatch,
    hydrogen_settings,
    expected_hydrogen_count,
):
    """3D、2D 与导出必须共用含已应用氢键结果的 context。"""

    class FakeStreamlit:
        def __init__(self):
            self.session_state = {}
            self.sidebar = self
            self.subheaders = []
            self.captions = []

        def expander(self, *_args, **_kwargs):
            return nullcontext()

        def spinner(self, *_args, **_kwargs):
            return nullcontext()

        def file_uploader(self, *_args, **_kwargs):
            return None

        def text_input(self, *_args, **_kwargs):
            return "app-context-test"

        def subheader(self, label, **_kwargs):
            self.subheaders.append(label)

        def caption(self, value, **_kwargs):
            self.captions.append(value)

        def __getattr__(self, _name):
            return lambda *_args, **_kwargs: None

    atoms = Atoms(
        "OHO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[10.0, 8.0, 8.0],
        pbc=[True, False, False],
    )
    default_style = SimpleNamespace(style=_portable_style_with_complete_palette())
    state = VisualizationState(
        style=replace(
            default_style.style,
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.2),),
                hydrogen_bonds=hydrogen_settings,
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=0,
                a=PeriodicRange(-1, 2),
            ),
        ),
        atom_selection=AtomSelectionSettings(),
    )
    active = ActiveWorkspace.from_upload(atoms, "periodic.xyz", b"periodic")
    applied_view = AppliedViewState(
        camera=CameraState(),
        rotation_matrix=np.asarray(rotate("90x"), dtype=float),
        event_id=None,
        view_revision="test-view",
    )
    captured = {}
    real_resolve_context = app_module.resolve_render_context

    def capture_resolve_context(resolved_atoms, resolved_state):
        context = real_resolve_context(resolved_atoms, resolved_state)
        captured["base_context"] = context
        return context

    def capture_render_2d(rendered_atoms, output_config, **kwargs):
        captured["rendered_atoms"] = rendered_atoms
        captured["output_config"] = output_config
        captured["render_context"] = kwargs.get("render_context")
        figure = plt.subplots()[0]
        captured["figure_2d"] = figure
        return figure

    def capture_create_3d_figure(*_args, **kwargs):
        captured["figure_3d_context"] = kwargs.get("render_context")
        captured["figure_3d_messages"] = kwargs.get("figure_messages")
        return object()

    def capture_atom_viewer(**kwargs):
        captured["viewer_locale"] = kwargs.get("locale")
        captured["viewer_messages"] = kwargs.get("messages")
        return None

    def capture_export_downloads(_container, figure, *_args):
        captured["export_figure"] = figure

    monkeypatch.setattr(app_module, "st", FakeStreamlit())
    monkeypatch.setattr(app_module, "load_default_style", lambda: default_style)
    monkeypatch.setattr(
        app_module,
        "_active_workspace_from_upload",
        lambda _uploaded: (active, False),
    )
    monkeypatch.setattr(
        app_module,
        "_reset_visual_state_for_structure",
        lambda *_args, **_kwargs: state,
    )
    monkeypatch.setattr(app_module, "_render_json_imports", lambda *_args: None)
    monkeypatch.setattr(
        app_module,
        "_render_global_forms",
        lambda *_args: (state, "app-context-test", nullcontext()),
    )
    monkeypatch.setattr(
        app_module,
        "_initialize_applied_view",
        lambda *_args: applied_view,
    )
    monkeypatch.setattr(app_module, "resolve_render_context", capture_resolve_context)
    monkeypatch.setattr(app_module, "create_3d_figure", capture_create_3d_figure)
    monkeypatch.setattr(app_module, "atom_viewer", capture_atom_viewer)
    monkeypatch.setattr(app_module, "_apply_viewer_event", lambda *_args: None)
    monkeypatch.setattr(app_module, "render_2d", capture_render_2d)
    monkeypatch.setattr(app_module, "render_preview_png", lambda *_args, **_kwargs: b"")
    monkeypatch.setattr(
        app_module,
        "preview_image_html",
        lambda _data, **_kwargs: "",
    )
    monkeypatch.setattr(app_module, "export_figure", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app_module, "_render_export_downloads", capture_export_downloads)
    monkeypatch.setattr(
        app_module,
        "_initialize_i18n",
        lambda: I18n(Locale.ZH_CN),
    )
    monkeypatch.setattr(
        app_module,
        "_render_locale_selector",
        lambda i18n: i18n,
    )

    app_module.main()

    base_context = captured["base_context"]
    output_context = captured["render_context"]
    assert captured["rendered_atoms"] is active.atoms
    assert output_context is not None
    assert output_context is not base_context
    assert captured["figure_3d_context"] is output_context
    assert captured["figure_3d_messages"]["bonds"] == "化学键"
    assert captured["viewer_locale"] is Locale.ZH_CN
    assert captured["viewer_messages"]["camera.apply"] == "应用当前视角"
    assert output_context.config is captured["output_config"]
    assert output_context.config is not base_context.config
    assert np.array_equal(
        output_context.config.rotation_matrix,
        applied_view.rotation_matrix,
    )
    assert output_context.bond_settings is base_context.bond_settings
    assert output_context.bond_resolution is base_context.bond_resolution
    assert output_context.periodic_topology_bonds is base_context.periodic_topology_bonds
    assert output_context.periodic_display is base_context.periodic_display
    assert output_context.hydrogen_bonds is base_context.hydrogen_bonds
    assert len(output_context.hydrogen_bonds) == expected_hydrogen_count
    assert output_context.hidden_atom_indices is base_context.hidden_atom_indices
    assert captured["export_figure"] is captured["figure_2d"]
    assert app_module.st.captions.count(THREE_D_INTERACTION_CAPTION) == 1
    assert all("氢键" not in caption for caption in app_module.st.captions)
    assert "3D 交互预览" in app_module.st.subheaders
    assert "📥 导出" not in app_module.st.subheaders


def test_cross_layer_periodic_identity_and_input_immutability(monkeypatch):
    """任一渲染层改用源行号或改写 Atoms 时，跨层身份集合必须失败。"""
    atoms, resolution = cross_layer_periodic_fixture()
    snapshot = (
        atoms.positions.copy(),
        atoms.cell.array.copy(),
        atoms.pbc.copy(),
        atoms.get_chemical_symbols(),
    )
    state = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                pair_rules=(
                    BondPairRule("C", "C", 0.0, 2.0),
                    BondPairRule("H", "O", 0.0, 1.3),
                ),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=2,
                a=PeriodicRange(-1, 2),
                b=PeriodicRange(0, 2),
            ),
        ),
        atom_selection=AtomSelectionSettings(
            hidden_atoms=(HiddenAtom(6, "C"),),
            color_strengths=(AtomColorStrength(11, "H", 0.30),),
        ),
    )
    monkeypatch.setattr(
        visual_state_module,
        "resolve_bonds",
        lambda *_args, **_kwargs: resolution,
    )

    context = resolve_render_context(atoms, state)
    _assert_atoms_unchanged(atoms, snapshot)
    assert context.periodic_display.base_image_shifts[3:7] == (
        (0, 0, -1),
        (0, 0, -1),
        (0, 0, -1),
        (0, 0, -1),
    )
    assert context.periodic_display.base_image_shifts[7] == (1, 0, 0)
    assert context.hydrogen_bonds
    assert {round(item.color_strength, 12) for item in context.hydrogen_bonds} == {
        0.30
    }

    expected = {
        (instance.source_atom_index, instance.image_shift)
        for instance in context.periodic_display.atom_instances
        if instance.source_atom_index not in context.hidden_atom_indices
    }
    figure_3d = create_3d_figure(
        atoms,
        context.config,
        render_context=context,
    )
    _assert_atoms_unchanged(atoms, snapshot)
    atom_trace = next(
        trace
        for trace in figure_3d.data
        if trace.meta["meia_role"] == "atoms"
    )
    plotly_ids = {
        (int(row[0]), tuple(int(value) for value in row[3]))
        for row in atom_trace.customdata
    }
    plotly_bond_rows = [
        row
        for trace in figure_3d.data
        if trace.meta
        and trace.meta.get("meia_role") in {"bonds", "bond_outlines"}
        for row in trace.customdata
        if row is not None
    ]
    assert plotly_bond_rows
    assert all(
        6 not in (int(row[1]), int(row[2]))
        for row in plotly_bond_rows
    )

    projection = projection_module.project_periodic_display(
        atoms,
        context.periodic_display,
        context.config,
        context.hidden_atom_indices,
    )
    _assert_atoms_unchanged(atoms, snapshot)
    projection_ids = {
        (int(source), tuple(int(value) for value in image_shift))
        for source, image_shift in zip(
            projection.source_atom_indices,
            projection.image_shifts,
        )
    }
    bond_geometries = compute_bond_geometries(
        context.periodic_display.bond_instances,
        projection,
        context.config,
    )
    assert bond_geometries
    assert all(
        6 not in (geometry.atom_i, geometry.atom_j)
        for geometry in bond_geometries
    )

    figure_2d = render_2d(atoms, context.config, render_context=context)
    svg = export_figure(figure_2d, "svg", context.config)
    _assert_atoms_unchanged(atoms, snapshot)
    svg_root = ET.fromstring(svg)
    svg_ids = {
        (
            int(node.attrib["data-meia-source-atom-index"]),
            tuple(
                int(value)
                for value in node.attrib["data-meia-image-shift"].split(",")
            ),
        )
        for node in svg_root.findall(
            ".//*[@data-meia-source-atom-index]"
        )
    }
    manifest_bonds = tuple(figure_2d._meia_bond_manifest.values())
    assert manifest_bonds
    assert all(
        6 not in (metadata["atom_i"], metadata["atom_j"])
        for metadata in manifest_bonds
    )
    svg_bond_groups = [
        node
        for node in svg_root.iter()
        if "data-atom-a" in node.attrib and "data-atom-b" in node.attrib
    ]
    assert svg_bond_groups
    assert all(
        6
        not in (
            int(group.attrib["data-atom-a"]),
            int(group.attrib["data-atom-b"]),
        )
        for group in svg_bond_groups
    )

    assert plotly_ids == projection_ids == svg_ids == expected
    assert all(source != 6 for source, _image_shift in expected)
    plt.close(figure_2d)


def test_all_hidden_atoms_keep_empty_layers_valid_exports_and_one_cell():
    """全隐藏时不得残留原子/键/氢键，也不得生成重复晶胞或空文件。"""
    atoms = Atoms(
        "OHO",
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        cell=[6.0, 5.0, 4.0],
        pbc=True,
    )
    state = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.2),),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=2,
                a=PeriodicRange(-1, 2),
            ),
        ),
        atom_selection=AtomSelectionSettings(
            hidden_atoms=(
                HiddenAtom(0, "O"),
                HiddenAtom(1, "H"),
                HiddenAtom(2, "O"),
            ),
        ),
    )
    context = resolve_render_context(atoms, state)

    figure_3d = create_3d_figure(
        atoms,
        context.config,
        render_context=context,
    )
    atom_trace = next(
        trace
        for trace in figure_3d.data
        if trace.meta and trace.meta.get("meia_role") == "atoms"
    )
    visible_roles = {
        trace.meta.get("meia_role")
        for trace in figure_3d.data
        if trace.meta
    }
    assert len(atom_trace.x) == 0
    assert visible_roles.isdisjoint(
        {"bonds", "bond_outlines", "hydrogen_bonds"}
    )
    assert [trace.name for trace in figure_3d.data].count("晶胞") == 1

    figure_2d = render_2d(atoms, context.config, render_context=context)
    assert figure_2d._meia_atom_manifest == {}
    assert figure_2d._meia_bond_manifest == {}
    assert figure_2d._meia_hydrogen_bond_manifest == {}
    assert not any(
        isinstance(patch, (Circle, Ellipse, Polygon))
        for patch in figure_2d.axes[0].patches
    )
    assert not any(
        isinstance(line, Line2D)
        and str(line.get_gid()).startswith("hydrogen_bond_")
        for line in figure_2d.axes[0].lines
    )

    svg = export_figure(figure_2d, "svg", context.config)
    png = export_figure(figure_2d, "png", context.config)
    pdf = export_figure(figure_2d, "pdf", context.config)
    assert svg.startswith(b"<?xml")
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.startswith(b"%PDF")
    svg_root = ET.fromstring(svg)
    assert not svg_root.findall(".//*[@data-meia-source-atom-index]")

    primary_state = replace(
        state,
        style=replace(
            state.style,
            cell_periodic=replace(
                state.style.cell_periodic,
                a=PeriodicRange(0, 1),
            ),
        ),
    )
    primary_context = resolve_render_context(atoms, primary_state)
    primary_figure = render_2d(
        atoms,
        primary_context.config,
        render_context=primary_context,
    )
    primary_svg_root = ET.fromstring(
        export_figure(primary_figure, "svg", primary_context.config)
    )
    svg_patch_ids = {
        node.attrib["id"]
        for node in svg_root.iter()
        if node.attrib.get("id", "").startswith("patch_")
    }
    primary_patch_ids = {
        node.attrib["id"]
        for node in primary_svg_root.iter()
        if node.attrib.get("id", "").startswith("patch_")
    }
    assert svg_patch_ids == primary_patch_ids
    assert svg_patch_ids
    assert not any(
        "translated_cell" in node.attrib.get("id", "")
        or "cell_replica" in node.attrib.get("id", "")
        for node in svg_root.iter()
    )
    plt.close(primary_figure)
    plt.close(figure_2d)


def test_loaded_app_reports_when_every_source_atom_is_hidden():
    """页面若在空原子层上仍显示普通预览提示，用户无法判断隐藏结果。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    atoms = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    active = ActiveWorkspace.from_upload(atoms, "all-hidden.xyz", b"all-hidden")
    preset = load_default_style()
    state = VisualizationState(
        style=preset.style,
        atom_selection=AtomSelectionSettings(
            hidden_atoms=(HiddenAtom(0, "H"), HiddenAtom(1, "O")),
        ),
    )
    app = _app_test(app_path)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.session_state[VISUAL_STRUCTURE_ID_KEY] = active.structure_id
    app.session_state["meia_visual_state"] = state

    app.run(timeout=30)

    assert not app.exception
    assert "当前没有可见原子" in [caption.value for caption in app.caption]


def test_new_snapshot_requires_a_fresh_overwrite_confirmation():
    """取消后或换文件时，不得沿用上一份快照的覆盖确认。"""
    state = {
        PENDING_SNAPSHOT_HASH_KEY: "first-payload",
        SNAPSHOT_CONFIRMATION_KEY: True,
    }

    changed = _reset_snapshot_confirmation_for_payload(
        state,
        "second-payload",
    )

    assert changed is True
    assert state[SNAPSHOT_CONFIRMATION_KEY] is False

    state[SNAPSHOT_CONFIRMATION_KEY] = True
    state[SNAPSHOT_CONFIRMATION_RESET_KEY] = True
    _consume_snapshot_confirmation_reset(state)
    assert state[SNAPSHOT_CONFIRMATION_KEY] is False
    assert SNAPSHOT_CONFIRMATION_RESET_KEY not in state


def test_three_d_caption_is_fixed_and_excludes_hydrogen_parameters():
    assert THREE_D_INTERACTION_CAPTION == (
        "拖拽调整视角；开启“选择模式”后可连续点击或拖框，点击“确认选择”才一次同步到侧栏，"
        "选择模式下滚轮仍可缩放。应用当前视角把 3D 相机传给 2D。"
    )
    assert "氢键" not in THREE_D_INTERACTION_CAPTION


def test_reset_cleanup_only_remains_in_visual_module_scope():
    session_state = {
        "meia_atom_cell_global_scale": 1,
        "meia_bond_form_draw": True,
        "meia_cell_periodic_a": 1,
        "meia_atom_selection_hidden": "1",
        "meia_processed_selection_event_id": "event",
        "meia_pending_atom_selection_indices": [1],
        "meia_export_form_format": "png",
        "meia_preset_name": "keep",
        "meia_applied_camera": "keep",
        PENDING_SNAPSHOT_KEY: "keep",
        RESET_STYLE_BASELINE_KEY: "keep",
    }

    _clear_reset_scoped_widgets(session_state)

    assert session_state == {
        "meia_export_form_format": "png",
        "meia_preset_name": "keep",
        "meia_applied_camera": "keep",
        PENDING_SNAPSHOT_KEY: "keep",
        RESET_STYLE_BASELINE_KEY: "keep",
    }


def test_reset_reinitialize_consumes_browser_rehydrated_visual_drafts():
    state = {
        RESET_WIDGET_REINITIALIZE_KEY: True,
        "meia_atom_cell_global_scale": 1.2,
        "meia_bond_form_draw": False,
        "meia_cell_periodic_a_end": 2,
        "meia_atom_selection_range": "1",
        "meia_export_form_format": "png",
        "meia_applied_camera": "keep",
    }

    assert _consume_reset_widget_reinitialize(state) is True
    assert state == {
        "meia_export_form_format": "png",
        "meia_applied_camera": "keep",
    }
    assert _consume_reset_widget_reinitialize(state) is False


def test_app_exposes_explicit_apply_button_for_each_global_form():
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    app = _app_test(app_path).run(timeout=30)

    button_labels = {button.label for button in app.button}
    assert button_labels >= {
        "应用原子设置",
        "应用晶胞与周期性设置",
        "应用导出设置",
    }
    all_visible_text = "\n".join(
        [caption.value for caption in app.caption]
        + [button.label for button in app.button]
    )
    assert "应用当前视角与原子样式" not in all_visible_text


def test_empty_app_sidebar_modules_are_collapsible():
    """JSON 导入操作应归入文件模块，不再占用独立折叠区。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    app = _app_test(app_path).run(timeout=30)

    expanders = [
        (expander.label, expander.proto.expanded) for expander in app.expander
    ]
    assert expanders == [
        ("📁 文件", True),
        ("原子", False),
        ("化学键", False),
        ("晶胞与周期性", False),
        ("导出", False),
    ]
    assert not app.expander[0].get("text_input")
    export_names = [item.label for item in app.expander[-1].get("text_input")]
    assert export_names == ["导出名称"]
    file_module_buttons = {
        button.label for button in app.expander[0].get("button")
    }
    assert file_module_buttons >= {
        "应用通用风格预设",
        "确认导入工作状态快照",
        "取消本次快照导入",
    }


def test_loaded_app_sidebar_order_and_downloads_are_complete():
    """载入结构后，七个折叠区的顺序与侧栏下载项必须完整。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms(
            "HO",
            positions=[[0, 0, 0], [1.0, 0, 0]],
            cell=[5, 5, 5],
            pbc=True,
        ),
        "fixture.xyz",
        b"fixture",
    )
    app = _app_test(app_path)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.run(timeout=30)

    assert not app.exception
    assert [expander.label for expander in app.expander] == [
        "📁 文件",
        "原子",
        "化学键",
        "晶胞与周期性",
        "原子选择",
        "导出",
    ]
    assert [button.label for button in app.get("download_button")] == [
        "下载 SVG",
        "下载通用风格预设",
        "下载工作状态快照",
    ]
    assert "📥 导出" not in [item.value for item in app.subheader]


def test_active_sidebar_uses_collapsible_selection_without_synthesizing_pairs(
    monkeypatch,
):
    """通用默认风格不在侧栏隐式合成任何元素对。"""

    class FakeSidebarStreamlit:
        def __init__(self):
            self.sidebar = nullcontext()
            self.session_state = {}
            self.expanders = []
            self.buttons = []
            self.download_container = nullcontext()

        def divider(self):
            return None

        def header(self, _label):
            return None

        def caption(self, _value):
            return None

        def expander(self, label, *, expanded=False):
            self.expanders.append((label, expanded))
            return nullcontext()

        def text_input(self, _label, *, value="", **_kwargs):
            return value

        def container(self):
            return self.download_container

        def button(self, label, *, disabled=False, **_kwargs):
            self.buttons.append((label, disabled))
            return False

    fake = FakeSidebarStreamlit()
    atoms = Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]])
    default_style = SimpleNamespace(style=PortableStyle())
    state = VisualizationState(style=default_style.style)
    received_pairs = []

    monkeypatch.setattr("app.st", fake)
    monkeypatch.setattr("app.render_atom_cell_form", lambda *_args: None)
    monkeypatch.setattr("app.render_bond_form", lambda *_args: None)
    monkeypatch.setattr("app.render_cell_periodic_form", lambda *_args: None)
    monkeypatch.setattr("app.render_export_form", lambda *_args: None)

    def capture_selection(_current, _atoms, available_pairs, _i18n):
        received_pairs.extend(available_pairs)
        return None

    monkeypatch.setattr("app.render_atom_selection_form", capture_selection)

    returned_state, export_name, download_container = _render_global_forms(
        SimpleNamespace(atoms=atoms), state, default_style
    )

    assert fake.expanders == [
        ("原子", False),
        ("化学键", False),
        ("晶胞与周期性", False),
        ("原子选择", False),
        ("导出", False),
    ]
    assert fake.buttons == [("一键还原初始配置", False)]
    assert received_pairs == []
    assert returned_state is state
    assert export_name == "meia-visual-state"
    assert download_container is fake.download_container


def test_atom_form_submit_reruns_to_refresh_scaled_element_radii(monkeypatch):
    """应用倍率后必须立即重跑，使元素绝对半径字段与已应用状态一致。"""

    class FakeSidebarStreamlit:
        def __init__(self):
            self.sidebar = nullcontext()
            self.session_state = {}
            self.rerun_count = 0

        def divider(self): return None
        def header(self, _label): return None
        def expander(self, _label, *, expanded=False): return nullcontext()
        def text_input(self, _label, *, value="", **_kwargs): return value
        def container(self): return nullcontext()
        def button(self, _label, **_kwargs): return False
        def rerun(self): self.rerun_count += 1

    fake = FakeSidebarStreamlit()
    atoms = Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]])
    current = VisualizationState(style=PortableStyle())
    scaled_atom_cell = replace(current.style.atom_cell, outline_width=0.7)
    scaled_profiles = replace(
        current.style.size_profiles,
        covalent=replace(
            current.style.size_profiles.covalent,
            global_scale=0.9,
        ),
    )
    monkeypatch.setattr("app.st", fake)
    monkeypatch.setattr(
        "app.render_atom_cell_form",
        lambda *_args: AtomFormSubmission(scaled_atom_cell, scaled_profiles),
    )
    monkeypatch.setattr("app.render_bond_form", lambda *_args: None)
    monkeypatch.setattr("app.render_cell_periodic_form", lambda *_args: None)
    monkeypatch.setattr("app.render_export_form", lambda *_args: None)
    monkeypatch.setattr("app.render_atom_selection_form", lambda *_args: None)

    returned, _name, _container = _render_global_forms(
        SimpleNamespace(atoms=atoms),
        current,
        SimpleNamespace(style=PortableStyle()),
    )

    assert returned.style.atom_cell == scaled_atom_cell
    assert returned.style.size_profiles == scaled_profiles
    assert (
        fake.session_state["meia_visual_state"].style.atom_cell
        == scaled_atom_cell
    )
    assert (
        fake.session_state["meia_visual_state"].style.size_profiles
        == scaled_profiles
    )
    assert fake.rerun_count == 1


def test_app_atom_scale_submit_recomputes_display_radius_fields():
    """真实 Streamlit 提交流程中，全局倍率应按当前比例更新全部绝对半径。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]]),
        "fixture.xyz",
        b"fixture",
    )
    app = _app_test(app_path)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.run(timeout=30)

    oxygen = next(
        item
        for item in app.number_input
        if item.label == "O 最终显示半径 / Å"
    )
    oxygen.set_value(0.8)
    next(item for item in app.button if item.label == "应用原子设置").click()
    app.run(timeout=30)

    next(item for item in app.slider if item.label == "全局半径缩放").set_value(0.9)
    next(item for item in app.button if item.label == "应用原子设置").click()
    app.run(timeout=30)

    radius_values = {item.label: item.value for item in app.number_input}
    assert radius_values["H 最终显示半径 / Å"] == pytest.approx(0.31 * 0.9)
    assert radius_values["O 最终显示半径 / Å"] == pytest.approx(0.8 * 1.5)
    applied = app.session_state[
        "meia_visual_state"
    ].style.size_profiles.covalent
    assert applied.global_scale == pytest.approx(0.9)
    assert applied.reference_overrides_angstrom["O"] == pytest.approx(
        0.8 / 0.6
    )


def test_app_uniform_base_can_decrease_repeatedly_without_locking_elements():
    """真实 Streamlit 重跑中，统一基础半径连续递减不得生成伪元素覆盖。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]]),
        "water.xyz",
        b"water",
    )
    app = _app_test(app_path)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.run(timeout=30)

    app.session_state["meia_atom_cell_radius_mode"] = "uniform"
    app.run(timeout=30)
    for reference_radius in (0.50, 0.40, 0.30):
        next(
            item for item in app.number_input if item.label == "统一基础半径 / Å"
        ).set_value(reference_radius)
        next(item for item in app.button if item.label == "应用原子设置").click()
        app.run(timeout=30)

    radius_values = {item.label: item.value for item in app.number_input}
    assert radius_values["H 最终显示半径 / Å"] == pytest.approx(0.30)
    assert radius_values["O 最终显示半径 / Å"] == pytest.approx(0.30)
    profile = app.session_state[
        "meia_visual_state"
    ].style.size_profiles.uniform
    assert profile.reference_radius_angstrom == pytest.approx(0.30)
    assert dict(profile.reference_overrides_angstrom) == {}


def test_app_mode_switch_applies_radius_and_matching_bond_width_atomically():
    """模式草稿不应抢先生效；应用后半径和对应键宽必须同时切换。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]]),
        "water.xyz",
        b"water",
    )
    default_style = load_default_style()
    base_state = app_module.initialize_visual_state(active.atoms, default_style)
    profiles = SizeProfileSettings(
        covalent=CovalentSizeProfile(
            global_scale=0.6,
            bond_width_ratio=0.31,
        ),
        uniform=UniformSizeProfile(
            global_scale=1.0,
            reference_radius_angstrom=0.35,
            bond_width_ratio=0.57,
        ),
    )
    state = replace(
        base_state,
        style=replace(base_state.style, size_profiles=profiles),
    )
    app = _app_test(app_path)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.session_state[VISUAL_STRUCTURE_ID_KEY] = active.structure_id
    app.session_state[VISUAL_STATE_KEY] = state
    app.run(timeout=30)

    app.session_state["meia_atom_cell_radius_mode"] = "uniform"
    app.run(timeout=30)

    draft_state = app.session_state[VISUAL_STATE_KEY]
    assert draft_state.style.size_profiles.active_mode is ProfileRadiusMode.COVALENT
    assert resolve_render_context(
        active.atoms,
        draft_state,
    ).config.bond_width_ratio == pytest.approx(0.31)

    next(item for item in app.button if item.label == "应用原子设置").click()
    app.run(timeout=30)

    applied_state = app.session_state[VISUAL_STATE_KEY]
    assert applied_state.style.size_profiles.active_mode is ProfileRadiusMode.UNIFORM
    context = resolve_render_context(active.atoms, applied_state)
    assert context.config.get_atom_radii(["H", "O"]) == pytest.approx([0.35, 0.35])
    assert context.config.bond_width_ratio == pytest.approx(0.57)


def test_app_mounts_large_imported_radius_across_scale_draft_rerun():
    """真实 Streamlit 控件应承接合法超范围状态并安全重算倍率草稿。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms("O", positions=[[0, 0, 0]]),
        "large-radius.xyz",
        b"large-radius",
    )
    default_style = load_default_style()
    base_state = app_module.initialize_visual_state(active.atoms, default_style)
    profiles = SizeProfileSettings(
        active_mode=ProfileRadiusMode.UNIFORM,
        uniform=UniformSizeProfile(
            global_scale=2.0,
            reference_radius_angstrom=10.0,
            reference_overrides_angstrom={"O": 12.0},
        ),
    )
    state = replace(
        base_state,
        style=replace(
            base_state.style,
            size_profiles=profiles,
        ),
    )
    app = _app_test(app_path)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.session_state[VISUAL_STRUCTURE_ID_KEY] = active.structure_id
    app.session_state[VISUAL_STATE_KEY] = state

    app.run(timeout=30)
    assert not app.exception
    radius_values = {item.label: item.value for item in app.number_input}
    assert radius_values["O 最终显示半径 / Å"] == pytest.approx(24.0)

    next(item for item in app.slider if item.label == "全局半径缩放").set_value(0.1)
    app.run(timeout=30)

    assert not app.exception
    radius_values = {item.label: item.value for item in app.number_input}
    assert radius_values["O 最终显示半径 / Å"] == pytest.approx(1.2)


def test_sidebar_reset_button_uses_successful_style_baseline_atomically(monkeypatch):
    """一键还原只恢复视觉模块，保留视角、导出与快照/导出会话状态。"""
    class FakeResetStreamlit:
        def __init__(self):
            self.sidebar = nullcontext()
            self.session_state = {}
        def divider(self): return None
        def header(self, _label): return None
        def caption(self, _value): return None
        def expander(self, label, *, expanded=False):
            return nullcontext()
        def text_input(self, _label, *, value="", **_kwargs): return value
        def container(self): return nullcontext()
        def button(self, label, **_kwargs): return label == "一键还原初始配置"
        def error(self, message): raise AssertionError(message)
        def rerun(self): self.rerun_count = getattr(self, "rerun_count", 0) + 1

    atoms = Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]])
    baseline = PortableStyle(
        atom_cell=replace(AtomCellSettings(), outline_width=0.75)
    )
    current = VisualizationState(
        style=replace(
            baseline,
            atom_cell=replace(baseline.atom_cell, outline_width=1.25),
            export=ExportSettings("png", 300, False),
        ),
        atom_selection=AtomSelectionSettings(selected_atom_indices=(1,)),
    )
    fake = FakeResetStreamlit()
    fake.session_state[RESET_STYLE_BASELINE_KEY] = baseline
    fake.session_state.update({
        "meia_atom_cell_global_scale": 1.2,
        "meia_export_form_format": "png",
        "meia_preset_name": "keep",
        PENDING_SNAPSHOT_KEY: "keep",
        RESET_STYLE_BASELINE_KEY: baseline,
    })
    monkeypatch.setattr("app.st", fake)
    monkeypatch.setattr("app.render_atom_cell_form", lambda *_args: None)
    monkeypatch.setattr("app.render_bond_form", lambda *_args: None)
    monkeypatch.setattr("app.render_cell_periodic_form", lambda *_args: None)
    monkeypatch.setattr("app.render_export_form", lambda *_args: None)
    monkeypatch.setattr("app.render_atom_selection_form", lambda *_args: None)

    returned_state, _export_name, _download_container = _render_global_forms(
        SimpleNamespace(atoms=atoms), current, SimpleNamespace(style=PortableStyle())
    )

    assert returned_state is current
    reset_state = fake.session_state["meia_visual_state"]
    assert reset_state.style.atom_cell == baseline.atom_cell
    assert reset_state.style.view is current.style.view
    assert reset_state.style.export == current.style.export
    assert reset_state.atom_selection == AtomSelectionSettings()
    assert "meia_atom_cell_global_scale" not in fake.session_state
    assert fake.session_state["meia_export_form_format"] == "png"
    assert fake.session_state["meia_preset_name"] == "keep"
    assert fake.session_state[PENDING_SNAPSHOT_KEY] == "keep"
    assert fake.session_state[RESET_STYLE_BASELINE_KEY] is baseline
    assert fake.rerun_count == 1


def test_app_reset_clears_browser_rehydrated_selection_draft():
    """还原后的重跑不得让浏览器把旧选择输入重新写回侧栏。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    active = ActiveWorkspace.from_upload(
        Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]]),
        "fixture.xyz",
        b"fixture",
    )
    app = _app_test(app_path)
    app.session_state[ACTIVE_WORKSPACE_KEY] = active
    app.run(timeout=30)

    next(
        item for item in app.text_input if item.label == "按序号加入选择"
    ).set_value("1")
    next(item for item in app.button if item.label == "一键还原初始配置").click()
    app.run(timeout=30)

    selection_range = next(
        item for item in app.text_input if item.label == "按序号加入选择"
    )
    assert selection_range.value == ""
    revision = app.session_state[ATOM_SELECTION_DRAFT_REVISION_KEY]
    assert app.session_state[
        atom_selection_draft_widget_key(
            "meia_atom_selection_range",
            revision,
        )
    ] == ""


def test_sidebar_reset_failure_preserves_session_atomically(monkeypatch):
    class FakeResetFailureStreamlit:
        def __init__(self, session_state):
            self.sidebar = nullcontext()
            self.session_state = session_state
            self.errors = []
            self.rerun_count = 0
        def divider(self): return None
        def header(self, _label): return None
        def expander(self, _label, *, expanded=False): return nullcontext()
        def text_input(self, _label, *, value="", **_kwargs): return value
        def container(self): return nullcontext()
        def button(self, label, **_kwargs): return label == "一键还原初始配置"
        def error(self, message): self.errors.append(message)
        def rerun(self): self.rerun_count += 1

    atoms = Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]])
    current = VisualizationState(PortableStyle())
    baseline = PortableStyle()
    sentinel = object()
    session_state = {
        "meia_visual_state": current,
        RESET_STYLE_BASELINE_KEY: baseline,
        "meia_atom_cell_global_scale": 1.2,
        "unrelated": sentinel,
    }
    before = dict(session_state)
    fake = FakeResetFailureStreamlit(session_state)
    monkeypatch.setattr(app_module, "st", fake)
    monkeypatch.setattr(app_module, "render_atom_cell_form", lambda *_args: None)
    monkeypatch.setattr(app_module, "render_bond_form", lambda *_args: None)
    monkeypatch.setattr(app_module, "render_cell_periodic_form", lambda *_args: None)
    monkeypatch.setattr(app_module, "render_export_form", lambda *_args: None)
    monkeypatch.setattr(app_module, "render_atom_selection_form", lambda *_args: None)
    monkeypatch.setattr(
        app_module,
        "reset_visual_modules_from_style",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid baseline")),
    )

    returned, _name, _container = _render_global_forms(
        SimpleNamespace(atoms=atoms),
        current,
        SimpleNamespace(style=PortableStyle()),
    )

    assert returned is current
    assert set(session_state) == set(before)
    for key, value in before.items():
        assert session_state[key] is value
    assert fake.errors == ["还原失败：ValueError: invalid baseline"]
    assert fake.rerun_count == 0


def test_sidebar_export_downloads_include_image_and_strict_v7_json(monkeypatch):
    """导出折叠区应在 2D 图完成后一次生成图像与两种 v7 JSON。"""

    class FakeDownloadStreamlit:
        def __init__(self):
            self.downloads = []
            self.errors = []

        def download_button(self, **kwargs):
            self.downloads.append(kwargs)

        def error(self, message):
            self.errors.append(message)

    atoms = Atoms("HO", positions=[[0, 0, 0], [1.0, 0, 0]])
    active = ActiveWorkspace.from_upload(atoms, "fixture.xyz", b"fixture")
    default_style = load_default_style().style
    state = VisualizationState(
        replace(
            default_style,
            bonds=replace(
                default_style.bonds,
                pair_rules=(
                    BondPairRule(
                        "H",
                        "O",
                        0.0,
                        1.2,
                        participates_in_periodic_unwrap=False,
                    ),
                ),
            ),
        )
    )
    fake = FakeDownloadStreamlit()
    figure = plt.subplots()[0]
    monkeypatch.setattr(app_module, "st", fake)
    monkeypatch.setattr(app_module, "export_figure", lambda *_args: b"<svg/>")

    _render_export_downloads(
        nullcontext(),
        figure,
        active,
        state,
        RenderConfig(),
        "   ",
    )

    assert [item["label"] for item in fake.downloads] == [
        "下载 SVG",
        "下载通用风格预设",
        "下载工作状态快照",
    ]
    for download in fake.downloads[1:]:
        assert b'"schema_version": 7' in download["data"]
        assert b'"size_profiles"' in download["data"]
        assert b'"hydrogen_bonds"' in download["data"]
        assert b'"participates_in_periodic_unwrap"' in download["data"]
    assert fake.downloads[1]["file_name"] == (
        "meia-visual-state.style.meia.json"
    )
    assert fake.downloads[2]["file_name"] == (
        "meia-visual-state.workspace.meia.json"
    )
    assert fake.errors == []
    plt.close(figure)


def test_sidebar_export_failure_stays_local_and_keeps_applied_state(monkeypatch):
    """图像导出失败只在侧栏报错，不得更换已应用状态。"""

    class FakeDownloadStreamlit:
        def __init__(self):
            self.errors = []

        def download_button(self, **_kwargs):
            raise AssertionError("导出失败时不应提供下载")

        def error(self, message):
            self.errors.append(message)

    atoms = Atoms("H", positions=[[0, 0, 0]])
    active = ActiveWorkspace.from_upload(atoms, "fixture.xyz", b"fixture")
    state = VisualizationState(load_default_style().style)
    fake = FakeDownloadStreamlit()
    figure = plt.subplots()[0]
    monkeypatch.setattr(app_module, "st", fake)

    def fail_export(*_args):
        raise ValueError("broken image")

    monkeypatch.setattr(app_module, "export_figure", fail_export)

    _render_export_downloads(
        nullcontext(),
        figure,
        active,
        state,
        RenderConfig(),
        "paper-style",
    )

    assert fake.errors == ["导出文件生成失败：ValueError: broken image"]
    assert state == VisualizationState(load_default_style().style)
    plt.close(figure)


def test_app_removes_sidebar_view_preset_but_keeps_z_up_default_state():
    """侧栏旧视角表单已删除，默认已应用视角仍为 -90x。"""
    app_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "app.py")
    app = _app_test(app_path).run(timeout=30)

    assert "视角预设" not in [selectbox.label for selectbox in app.selectbox]
    assert load_default_style().style.view.rotation == "-90x"


def test_camera_to_rotation_matrix():
    """验证 Plotly 相机参数 → 旋转矩阵转换。"""
    # 默认相机视角
    camera = {
        "eye": {"x": 1.25, "y": 1.25, "z": 1.25},
        "center": {"x": 0, "y": 0, "z": 0},
        "up": {"x": 0, "y": 0, "z": 1},
    }
    rm = camera_to_rotation_matrix(camera)
    assert rm.shape == (3, 3), f"旋转矩阵形状错误: {rm.shape}"

    # 验证正交性: R @ R.T ≈ I
    identity = rm @ rm.T
    err = np.abs(identity - np.eye(3)).max()
    assert err < 1e-10, f"旋转矩阵不正交，误差: {err}"

    # 验证行列式 = 1（右手系）
    det = np.linalg.det(rm)
    assert abs(det - 1.0) < 1e-10, f"行列式不为 1: {det}"

    print(f"\n[相机→旋转矩阵] 正交性误差: {err:.2e}, 行列式: {det:.6f}")


def test_camera_rotation_matrix_projects_onto_camera_basis():
    """行向量乘旋转矩阵后，应得到相机的右、上、深度坐标。"""
    camera = {
        "eye": {"x": 1.0, "y": 1.0, "z": 0.0},
        "center": {"x": 0.0, "y": 0.0, "z": 0.0},
        "up": {"x": 0.0, "y": 0.0, "z": 1.0},
    }
    rotation = camera_to_rotation_matrix(camera)
    root_two = np.sqrt(2.0)
    camera_right = np.array([-1.0 / root_two, 1.0 / root_two, 0.0])
    camera_up = np.array([0.0, 0.0, 1.0])
    camera_depth = np.array([1.0 / root_two, 1.0 / root_two, 0.0])

    assert np.allclose(camera_right @ rotation, [1.0, 0.0, 0.0])
    assert np.allclose(camera_up @ rotation, [0.0, 1.0, 0.0])
    assert np.allclose(camera_depth @ rotation, [0.0, 0.0, 1.0])


def test_render_2d(sample_atoms):
    """验证 2D 渲染函数（含旋转矩阵直接传入）。"""
    atoms = sample_atoms
    config = RenderConfig(rotation="90x")

    fig = render_2d(atoms, config)
    assert fig is not None
    assert len(fig.axes) > 0

    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"\n[2D 渲染] 通过，{len(atoms)} 个原子")


def test_render_2d_can_disable_bonds():
    """UI 关闭化学键时，2D 渲染不应创建键矩形或椭圆帽。"""
    atoms = Atoms("HO", positions=[[0.0, 0.0, 0.0], [0.90, 0.0, 0.0]])
    config = RenderConfig(show_unit_cell=0)

    fig = render_2d(atoms, config, draw_bonds=False)
    bond_patches = [
        patch
        for patch in fig.axes[0].patches
        if isinstance(patch, Polygon) or type(patch) is Ellipse
    ]

    assert bond_patches == []
    import matplotlib.pyplot as plt
    plt.close(fig)


def test_create_3d_figure_can_disable_bonds():
    """同一个 UI 开关也必须移除 3D 预览中的化学键 trace。"""
    atoms = Atoms("HO", positions=[[0.0, 0.0, 0.0], [0.90, 0.0, 0.0]])

    fig = create_3d_figure(atoms, RenderConfig(), draw_bonds=False)

    assert [trace.name for trace in fig.data] == ["原子"]


def test_same_pair_rule_hides_bonds_in_2d_and_3d():
    """2D 与 3D 必须读取同一份元素对规则，不能各自重新判断。"""
    atoms = Atoms("CaO", positions=[[0.0, 0.0, 0.0], [2.30, 0.0, 0.0]])
    settings = BondSettings(
        pair_rules=(BondPairRule("Ca", "O", 2.1, 2.8, enabled=False),),
    )

    fig_2d = render_2d(atoms, RenderConfig(show_unit_cell=0), bond_settings=settings)
    bond_patches = [
        patch
        for patch in fig_2d.axes[0].patches
        if isinstance(patch, Polygon) or type(patch) is Ellipse
    ]
    fig_3d = create_3d_figure(
        atoms,
        RenderConfig(show_unit_cell=0),
        bond_settings=settings,
    )

    assert bond_patches == []
    assert [trace.name for trace in fig_3d.data] == ["原子"]
    import matplotlib.pyplot as plt
    plt.close(fig_2d)


def test_2d_and_3d_both_use_the_unwrapped_periodic_bond_instance():
    """3D 接入周期显示图后，也必须使用展开后的跨边界键端点。"""
    atoms = Atoms(
        "OSi",
        positions=[[0.0, 0.0, 19.5], [0.0, 0.0, 1.1]],
        cell=[10.0, 10.0, 20.0],
        pbc=True,
    )
    settings = BondSettings(
        pair_rules=(BondPairRule("O", "Si", 1.5, 1.7, enabled=True),),
    )
    config = RenderConfig(show_unit_cell=0)

    fig_2d = render_2d(atoms, config, bond_settings=settings)
    fig_3d = create_3d_figure(atoms, config, bond_settings=settings)
    bond_patches = [
        patch
        for patch in fig_2d.axes[0].patches
        if isinstance(patch, Polygon) or type(patch) is Ellipse
    ]

    assert len(bond_patches) == 4
    bond_traces = [
        trace for trace in fig_3d.data
        if trace.meta and trace.meta.get("meia_role") == "bonds"
    ]
    assert len(bond_traces) == 2
    atom_trace = next(
        trace for trace in fig_3d.data
        if trace.meta and trace.meta.get("meia_role") == "atoms"
    )
    assert list(atom_trace.z) == pytest.approx([-0.5, 1.1])
    assert sorted(
        float(value)
        for trace in bond_traces
        for value in trace.z
        if value is not None
    ) == pytest.approx([-0.104, 0.165, 0.165, 0.434])
    import matplotlib.pyplot as plt

    plt.close(fig_2d)


def test_direct_2d_keeps_bond_settings_global_switch_off():
    """默认 draw_bonds 参数不得反向开启 BondSettings 的全局关闭状态。"""
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    settings = BondSettings(
        draw_bonds=False,
        pair_rules=(BondPairRule("C", "O", 1.0, 1.4),),
    )

    figure = render_2d(
        atoms,
        RenderConfig(show_unit_cell=0),
        bond_settings=settings,
    )
    bond_patches = [
        patch
        for patch in figure.axes[0].patches
        if isinstance(patch, Polygon) or type(patch) is Ellipse
    ]

    assert bond_patches == []

    import matplotlib.pyplot as plt

    plt.close(figure)


def test_create_3d_figure_uses_clipped_two_color_bond_halves():
    """3D 预览不得继续用一根中心到中心的单色线表示化学键。"""
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]])

    fig = create_3d_figure(
        atoms,
        RenderConfig(radius_scale=0.6, show_unit_cell=0),
    )

    bond_traces = [trace for trace in fig.data if trace.name == "化学键"]
    assert len(bond_traces) == 2
    traces_by_color = {trace.line.color: trace for trace in bond_traces}

    carbon_half = traces_by_color["#3F4F6A"]
    oxygen_half = traces_by_color["#E5A6A6"]
    assert np.allclose(carbon_half.x[:2], [0.456, 0.630])
    assert np.allclose(oxygen_half.x[:2], [0.630, 0.804])


def test_create_3d_figure_uses_095_opacity_atoms_with_1px_black_outlines():
    """普通原子使用 0.95 不透明度与统一 1 px 黑色轮廓。"""
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]])

    fig = create_3d_figure(
        atoms,
        RenderConfig(radius_scale=0.6, show_unit_cell=0),
    )

    atom_trace = next(trace for trace in fig.data if trace.name == "原子")
    assert atom_trace.marker.opacity == pytest.approx(0.95)
    assert atom_trace.marker.line.width == pytest.approx(1.0)
    assert atom_trace.marker.line.color == "#000000"


def test_create_3d_figure_shows_draft_strength_and_multiple_selected_atoms():
    """3D 草稿应同步填充/描边色，并能同时高亮批量选择。"""
    atoms = Atoms("HOC", positions=[[0, 0, 0], [0.9, 0, 0], [3, 0, 0]])
    config = RenderConfig(
        show_unit_cell=0,
        custom_colors={"H": "#E6E6E5", "O": "#DFA3A3", "C": "#3E4E68"},
        atom_color_strengths={0: 0.30, 1: 0.30},
    )

    fig = create_3d_figure(
        atoms,
        config,
        selected_atom_indices=(0, 1),
    )

    atom_trace = next(trace for trace in fig.data if trace.name == "原子")
    assert list(atom_trace.marker.color) == ["#F7F7F7", "#F5E3E3", "#3E4E68"]
    assert atom_trace.marker.line.color == "#000000"
    selected_trace = next(trace for trace in fig.data if trace.name == "批量选择")
    assert len(selected_trace.x) == 3
    assert selected_trace.meta["meia_role"] == "selection"
    assert list(selected_trace.marker.size[:2]) == list(atom_trace.marker.size[:2])
    assert selected_trace.marker.size[2] == 0
    assert [list(row[:2]) for row in selected_trace.customdata] == [
        [0, "H"],
        [1, "O"],
        [2, "C"],
    ]
    assert all(
        [list(row[2]), list(row[3])] == [[0, 0, 0], [0, 0, 0]]
        for row in selected_trace.customdata
    )
    assert list(selected_trace.marker.color) == [
        "rgba(255,213,79,0.55)",
        "rgba(255,213,79,0.55)",
        "rgba(0,0,0,0)",
    ]

    bond_colors = {
        trace.line.color
        for trace in fig.data
        if trace.name.startswith("化学键")
        and trace.name != "化学键描边"
    }
    assert bond_colors == {"#F7F7F7", "#F5E3E3"}
    bond_outlines = [
        trace
        for trace in fig.data
        if trace.meta and trace.meta.get("meia_role") == "bond_outlines"
    ]
    bond_fills = [
        trace
        for trace in fig.data
        if trace.meta and trace.meta.get("meia_role") == "bonds"
    ]
    assert {trace.line.color for trace in bond_outlines} == {
        "rgba(136,136,136,0.72)",
        "rgba(135,125,125,0.72)",
    }
    assert len(bond_outlines) == len(bond_fills) == 2
    assert all(
        outline.line.width == pytest.approx(fill.line.width + 1.0)
        for outline, fill in zip(bond_outlines, bond_fills)
    )


def test_create_3d_figure_renders_replicas_and_selects_by_source_identity():
    """回退到源坐标或按显示行选择时，第二副本与同步高亮必须失败。"""
    atoms = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=[True, False, False],
    )
    state = VisualizationState(
        style=PortableStyle(
            size_profiles=SizeProfileSettings(
                covalent=CovalentSizeProfile(global_scale=0.5)
            ),
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.0),),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=0,
                a=PeriodicRange(0, 2),
            ),
        ),
    )
    context = resolve_render_context(atoms, state)

    fig = create_3d_figure(
        atoms,
        context.config,
        selected_atom_indices=(0,),
        render_context=context,
    )

    atom_trace = next(
        trace for trace in fig.data
        if trace.meta["meia_role"] == "atoms"
    )
    assert atom_trace.meta["meia_source_atom_indices"] == [0, 1, 0, 1]
    assert list(atom_trace.x) == pytest.approx([0.0, 0.9, 4.0, 4.9])
    assert list(atom_trace.customdata[2][0:2]) == [0, "H"]
    assert list(atom_trace.customdata[2][2]) == [1, 0, 0]
    assert list(atom_trace.customdata[2][3]) == [1, 0, 0]
    assert "显示像 (1,0,0)" in atom_trace.text[2]

    selection = next(
        trace for trace in fig.data
        if trace.meta["meia_role"] == "selection"
    )
    selected_rows = [
        index for index, size in enumerate(selection.marker.size) if size > 0
    ]
    assert selected_rows == [0, 2]
    assert selection.meta["meia_source_atom_indices"] == [0, 1, 0, 1]

    bond_fills = {
        trace.line.color: trace
        for trace in fig.data
        if trace.meta and trace.meta.get("meia_role") == "bonds"
    }
    hydrogen_x = list(bond_fills["#E6E6E5"].x)
    oxygen_x = list(bond_fills["#E5A6A6"].x)
    assert [value for value in hydrogen_x if value is not None] == pytest.approx(
        [0.155, 0.3625, 4.155, 4.3625],
    )
    assert [value for value in oxygen_x if value is not None] == pytest.approx(
        [0.3625, 0.57, 4.3625, 4.57],
    )
    assert [index for index, value in enumerate(hydrogen_x) if value is None] == [2, 5]
    assert [index for index, value in enumerate(oxygen_x) if value is None] == [2, 5]


def test_create_3d_figure_hover_reports_actual_unwrapped_image_shift():
    atoms = Atoms(
        "HO",
        scaled_positions=[[0.02, 0.5, 0.5], [0.98, 0.5, 0.5]],
        cell=[10.0, 10.0, 10.0],
        pbc=[True, False, False],
    )
    state = VisualizationState(
        style=PortableStyle(
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.1, 1.0),),
            ),
            cell_periodic=CellPeriodicSettings(show_unit_cell=0),
        )
    )
    context = resolve_render_context(atoms, state)
    assert context.periodic_display.atom_instances[0].replica_translation == (0, 0, 0)
    assert context.periodic_display.atom_instances[0].image_shift == (1, 0, 0)

    figure = create_3d_figure(atoms, context.config, render_context=context)
    atom_trace = next(
        trace for trace in figure.data
        if trace.meta["meia_role"] == "atoms"
    )

    assert "显示像 (1,0,0)" in atom_trace.text[0]


def test_create_3d_figure_hides_every_replica_and_its_bonds():
    """隐藏源原子后，任何副本或关联周期键残留都必须失败。"""
    atoms = Atoms(
        "HO",
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]],
        cell=[4.0, 4.0, 4.0],
        pbc=[True, False, False],
    )
    state = VisualizationState(
        style=PortableStyle(
            size_profiles=SizeProfileSettings(
                covalent=CovalentSizeProfile(global_scale=0.5),
            ),
            bonds=BondModuleSettings(
                pair_rules=(BondPairRule("H", "O", 0.8, 1.0),),
            ),
            cell_periodic=CellPeriodicSettings(
                show_unit_cell=0,
                a=PeriodicRange(0, 2),
            ),
        ),
        atom_selection=AtomSelectionSettings(
            hidden_atoms=(HiddenAtom(0, "H"),),
        ),
    )
    context = resolve_render_context(atoms, state)

    fig = create_3d_figure(
        atoms,
        context.config,
        render_context=context,
    )

    atom_trace = next(
        trace for trace in fig.data
        if trace.meta["meia_role"] == "atoms"
    )
    assert atom_trace.meta["meia_source_atom_indices"] == [1, 1]
    assert list(atom_trace.x) == pytest.approx([0.9, 4.9])
    assert not any(
        trace.meta
        and trace.meta.get("meia_role") in {"bonds", "bond_outlines"}
        for trace in fig.data
    )


def test_create_3d_figure_uses_applied_orthographic_camera(sample_atoms):
    """3D figure 必须显式使用已应用相机和对应的 view revision。"""
    camera = CameraState(eye=(0.0, 2.0, 0.0))

    fig = create_3d_figure(
        sample_atoms,
        RenderConfig(),
        camera=camera,
        uirevision="preset:structure-1:90x",
    )

    assert fig.layout.scene.camera.eye.y == 2.0
    assert fig.layout.scene.camera.projection.type == "orthographic"
    assert fig.layout.scene.uirevision == "preset:structure-1:90x"


def test_create_3d_figure_respects_hidden_unit_cell(sample_atoms):
    """晶胞图层预设为隐藏时，3D 也不应继续添加晶胞 trace。"""
    fig = create_3d_figure(sample_atoms, RenderConfig(show_unit_cell=0))

    assert "晶胞" not in [trace.name for trace in fig.data]


def test_render_2d_with_rotation_matrix(sample_atoms):
    """验证直接传入旋转矩阵的 2D 渲染。"""
    atoms = sample_atoms
    config = RenderConfig(rotation="90x")

    # 先用字符串旋转获取参考结果
    proj_ref = project_atoms(atoms, config)

    # 用旋转矩阵传入
    config2 = RenderConfig(rotation_matrix=proj_ref.rotation_matrix)
    proj2 = project_atoms(atoms, config2)

    # 两者结果应一致
    err = np.abs(proj_ref.positions_2d - proj2.positions_2d).max()
    assert err < 1e-10, f"旋转矩阵传入结果与字符串不一致: {err}"
    print(f"\n[旋转矩阵传入] 与字符串旋转结果一致，误差: {err:.2e}")


def test_batch_process(sample_atoms):
    """验证批量处理脚本。"""
    atoms = sample_atoms
    tmp_dir = tempfile.mkdtemp(prefix="meia_batch_test_")
    out_dir = tempfile.mkdtemp(prefix="meia_batch_out_")

    try:
        # 写入两个测试文件
        from ase.io import write
        write(os.path.join(tmp_dir, "POSCAR"), atoms, format="vasp")
        write(os.path.join(tmp_dir, "test2.xyz"), atoms)

        # 扫描文件
        files = find_structure_files(tmp_dir)
        assert len(files) == 2, f"应找到 2 个文件，找到 {len(files)}"

        # 批量处理
        config = RenderConfig(rotation="90x", dpi=150)
        results = batch_process(tmp_dir, out_dir, config=config, output_format="svg")

        assert len(results) == 2
        for r in results:
            assert r is not None, "批量处理结果中有 None"
            assert os.path.exists(r), f"输出文件不存在: {r}"

        # 检查文件大小
        for r in results:
            size = os.path.getsize(r)
            assert size > 0, f"输出文件为空: {r}"

        print(f"\n[批量处理] 2 个文件全部成功，输出目录: {out_dir}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_batch_no_bonds(sample_atoms):
    """验证批量处理不绘制化学键的选项。"""
    atoms = sample_atoms
    tmp_dir = tempfile.mkdtemp(prefix="meia_batch_nb_")
    out_dir = tempfile.mkdtemp(prefix="meia_batch_nb_out_")

    try:
        from ase.io import write
        write(os.path.join(tmp_dir, "test.vasp"), atoms)

        config = RenderConfig(rotation="90x", dpi=150)
        results = batch_process(tmp_dir, out_dir, config=config, draw_bonds=False, output_format="png")

        assert len(results) == 1
        assert results[0] is not None
        assert os.path.exists(results[0])

        size = os.path.getsize(results[0])
        assert size > 0

        print(f"\n[批量处理-无键] PNG 输出成功，大小: {size} bytes")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        shutil.rmtree(out_dir, ignore_errors=True)


def test_batch_without_explicit_config_uses_complete_builtin_palette(tmp_path):
    """未传入配置或预设时，批处理应与网页读取同一份内置风格。"""
    from ase.io import write

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    write(input_dir / "ne.xyz", Atoms("Ne", positions=[[0, 0, 0]]))

    result = batch_process(
        str(input_dir),
        str(output_dir),
        draw_bonds=False,
        output_format="svg",
    )

    assert result == [str(output_dir / "ne.svg")]
    svg = (output_dir / "ne.svg").read_text(encoding="utf-8").lower()
    expected = load_default_style().style.atom_cell.element_colors["Ne"].lower()
    assert f"fill: {expected}" in svg


def test_batch_size_overrides_change_only_the_active_profile():
    profiles = SizeProfileSettings(
        active_mode=ProfileRadiusMode.UNIFORM,
        covalent=CovalentSizeProfile(
            global_scale=0.75,
            reference_overrides_angstrom={"H": 0.4, "O": 0.8},
            bond_width_ratio=0.31,
        ),
        uniform=UniformSizeProfile(
            global_scale=0.75,
            reference_radius_angstrom=1.2,
            reference_overrides_angstrom={"O": 1.3, "Si": 1.6},
            bond_width_ratio=0.57,
        ),
    )
    style = replace(
        _portable_style_with_complete_palette(),
        size_profiles=profiles,
    )
    preset = build_style_preset(VisualizationState(style), "batch", "0.11.0")

    updated = _style_preset_with_overrides(
        preset,
        radius_scale=1.1,
        bond_width_ratio=0.42,
    )

    assert updated.style.size_profiles.covalent == profiles.covalent
    assert updated.style.size_profiles.uniform == replace(
        profiles.uniform,
        global_scale=1.1,
        bond_width_ratio=0.42,
    )


def test_batch_cli_without_preset_only_overrides_explicit_style_flags(
    monkeypatch,
    tmp_path,
):
    """CLI 未给出的选项应继承内置 v3 风格，不应回落硬编码旧默认。"""
    import meia.batch as batch_module

    captured = {}

    def fake_batch_process(**kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(batch_module, "batch_process", fake_batch_process)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "meia.batch",
            str(tmp_path / "input"),
            "-o",
            str(tmp_path / "output"),
            "--rotation",
            "90x",
            "--format",
            "png",
        ],
    )

    batch_main()

    preset = captured["visualization_preset"]
    builtin = load_default_style()
    assert preset.style.view.rotation == "90x"
    assert np.allclose(
        camera_to_rotation_matrix(preset.style.view.camera),
        rotate("90x"),
    )
    assert preset.style.export.format == "png"
    assert preset.style.atom_cell == builtin.style.atom_cell
    assert preset.style.bonds == builtin.style.bonds
    assert preset.style.export.dpi == builtin.style.export.dpi
    assert preset.style.export.transparent == builtin.style.export.transparent
    assert captured["config"] is None


def test_batch_rejects_same_stem_output_collision_before_rendering(tmp_path):
    """两种输入格式同名时不得静默覆盖同一个导出文件。"""
    from ase.io import write

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    atoms = Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]], cell=[5, 5, 5])
    write(input_dir / "same.xyz", atoms)
    write(input_dir / "same.cif", atoms)

    with pytest.raises(ValueError, match="same\\.svg"):
        batch_process(str(input_dir), str(output_dir), output_format="svg")

    assert not output_dir.exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_batch_rejects_case_only_output_collision_even_with_overwrite(
    tmp_path,
    overwrite,
):
    """macOS 常见文件系统上仅大小写不同的目标仍是冲突。"""
    from ase.io import write

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    atoms = Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]], cell=[5, 5, 5])
    write(input_dir / "same.xyz", atoms)
    write(input_dir / "SAME.cif", atoms)

    with pytest.raises(ValueError, match="同名冲突"):
        batch_process(
            str(input_dir),
            str(output_dir),
            output_format="svg",
            overwrite=overwrite,
        )

    assert not output_dir.exists()


def test_batch_refuses_existing_output_unless_overwrite_is_explicit(tmp_path):
    """默认保护已有结果，仅显式 overwrite 才允许更新。"""
    from ase.io import write

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    write(
        input_dir / "structure.xyz",
        Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]]),
    )
    target = output_dir / "structure.svg"
    target.write_text("keep-me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="structure\\.svg"):
        batch_process(str(input_dir), str(output_dir), output_format="svg")
    assert target.read_text(encoding="utf-8") == "keep-me"

    results = batch_process(
        str(input_dir),
        str(output_dir),
        output_format="svg",
        overwrite=True,
    )
    assert results == [str(target)]
    assert target.read_bytes().startswith(b"<?xml")


def _reference_manifest_mapping():
    return {
        "schema_version": 2,
        "source_date_epoch": 1787241600,
        "created_at": "2026-08-23T00:00:00+08:00",
        "input": {
            "content_sha256": (
                "187ee6a6d1c5bffc2b55a8ea254f0dc86c82a1f56743fcdad55504488b399d5f"
            ),
            "atom_count": 225,
            "symbols_sha256": (
                "4555e45eecaefcfc308f5c386b206bf0b01e398dddfeefa57645f258ffc90a2d"
            ),
        },
        "view_rotation": "-90z,-90x",
        "color_strengths": [],
        "workspace": "examples/co2_h2o_color_strength.meia.json",
        "workspace_sha256": "",
        "output": "examples/co2_h2o_color_strength.svg",
        "output_sha256": "",
    }


@pytest.mark.release
@pytest.mark.parametrize(
    ("field", "edited_value"),
    [
        ("content_sha256", "0" * 64),
        ("atom_count", 3),
        ("symbols_sha256", "1" * 64),
        ("created_at", "1970-01-01T00:00:00+00:00"),
        ("view_rotation", "-90x"),
        ("workspace", "examples/edited.meia.json"),
        ("output", "examples/edited.svg"),
    ],
)
def test_reference_manifest_gate_rejects_each_code_owned_identity_edit(
    field,
    edited_value,
):
    manifest = _reference_manifest_mapping()
    if field in manifest["input"]:
        manifest["input"][field] = edited_value
    else:
        manifest[field] = edited_value

    with pytest.raises(ValueError, match="身份绑定"):
        regeneration_script._validate_identity_bound_manifest(
            regeneration_script.REFERENCE_MANIFEST_PATH,
            manifest,
        )


@pytest.mark.release
def test_reference_regeneration_rejects_substituted_source_without_writes(
    tmp_path,
):
    project_root = Path(__file__).resolve().parents[1]
    sandbox_root = tmp_path / "sandbox-project"
    sandbox_script = sandbox_root / "scripts" / "regenerate_visualization_example.py"
    sandbox_examples = sandbox_root / "examples"
    sandbox_script.parent.mkdir(parents=True)
    sandbox_examples.mkdir()
    shutil.copy2(
        project_root / "scripts" / "regenerate_visualization_example.py",
        sandbox_script,
    )

    input_path = tmp_path / "substituted.xyz"
    input_path.write_text(
        "3\nsubstituted source\nH 0 0 0\nO 0.9 0 0\nSi 2.5 0 0\n",
        encoding="utf-8",
    )
    content = input_path.read_bytes()
    manifest = _reference_manifest_mapping()
    manifest["input"] = {
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "atom_count": 3,
        "symbols_sha256": hashlib.sha256(b"H\0O\0Si").hexdigest(),
    }
    manifest["color_strengths"] = [
        {"atom_index": 1, "atom_symbol": "O", "strength": 0.4}
    ]
    manifest_path = sandbox_examples / "co2_h2o_color_strength.manifest.json"
    workspace_path = sandbox_examples / "co2_h2o_color_strength.meia.json"
    output_path = sandbox_examples / "co2_h2o_color_strength.svg"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    workspace_path.write_bytes(b"workspace-sentinel")
    output_path.write_bytes(b"svg-sentinel")

    def file_identity(path):
        stat = path.stat()
        return path.read_bytes(), stat.st_ino, stat.st_mtime_ns

    protected_paths = (manifest_path, workspace_path, output_path)
    before = {path: file_identity(path) for path in protected_paths}
    result = subprocess.run(
        [
            sys.executable,
            str(sandbox_script),
            "--input",
            str(input_path),
            "--manifest",
            str(manifest_path),
            "--overwrite",
        ],
        cwd=project_root,
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(
                filter(None, (str(project_root), os.environ.get("PYTHONPATH")))
            ),
        },
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "身份绑定" in result.stdout + result.stderr
    assert {path: file_identity(path) for path in protected_paths} == before


@pytest.mark.release
def test_regeneration_script_writes_and_checks_manifest_hashes(tmp_path):
    """临时构型的工作区、SVG 与清单哈希必须可逐字节再生。"""
    project_root = Path(__file__).resolve().parents[1]
    input_path = tmp_path / "three-atoms.xyz"
    input_path.write_text(
        "3\nthree atom fixture\nH 0 0 0\nO 0.9 0 0\nSi 2.5 0 0\n",
        encoding="utf-8",
    )
    content = input_path.read_bytes()
    source_stat = input_path.stat()
    source_identity = (
        hashlib.sha256(content).hexdigest(),
        source_stat.st_size,
        source_stat.st_mtime_ns,
        source_stat.st_ino,
    )
    with tempfile.TemporaryDirectory(
        prefix="meia-regeneration-",
        dir=project_root,
    ) as generated_dir:
        generated_dir = Path(generated_dir)
        workspace_path = generated_dir / "generated.meia.json"
        output_path = generated_dir / "generated.svg"
        manifest_path = generated_dir / "driver.manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "source_date_epoch": 0,
                    "created_at": "1970-01-01T00:00:00+00:00",
                    "input": {
                        "content_sha256": hashlib.sha256(content).hexdigest(),
                        "atom_count": 3,
                        "symbols_sha256": hashlib.sha256(b"H\0O\0Si").hexdigest(),
                    },
                    "view_rotation": "-90x",
                    "color_strengths": [
                        {
                            "atom_index": 1,
                            "atom_symbol": "O",
                            "strength": 0.4,
                        }
                    ],
                    "workspace": workspace_path.relative_to(project_root).as_posix(),
                    "workspace_sha256": "",
                    "output": output_path.relative_to(project_root).as_posix(),
                    "output_sha256": "",
                }
            ),
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(project_root / "scripts" / "regenerate_visualization_example.py"),
            "--input",
            str(input_path),
            "--manifest",
            str(manifest_path),
        ]

        overwritten = subprocess.run(
            command + ["--overwrite"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        assert overwritten.returncode == 0, overwritten.stderr
        current_source_stat = input_path.stat()
        assert (
            hashlib.sha256(input_path.read_bytes()).hexdigest(),
            current_source_stat.st_size,
            current_source_stat.st_mtime_ns,
            current_source_stat.st_ino,
        ) == source_identity
        generated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert generated_manifest["schema_version"] == 2
        assert generated_manifest["workspace_sha256"] == hashlib.sha256(
            workspace_path.read_bytes()
        ).hexdigest()
        assert generated_manifest["output_sha256"] == hashlib.sha256(
            output_path.read_bytes()
        ).hexdigest()
        generated_workspace = parse_preset(workspace_path.read_bytes())
        assert generated_workspace.metadata.schema_version == 7
        assert generated_workspace.metadata.meia_version == "0.11.0"
        assert generated_workspace.state.style.size_profiles == SizeProfileSettings()
        assert generated_workspace.metadata.name == "generated-reference"
        assert len(generated_workspace.structure.symbols) == 3
        assert generated_workspace.state.atom_selection.color_strengths == (
            AtomColorStrength(1, "O", 0.4),
        )
        assert {
            rule.pair for rule in generated_workspace.state.style.bonds.pair_rules
        } == {("H", "O"), ("O", "Si")}
        assert not any(
            "C" in rule.pair or "Ca" in rule.pair
            for rule in generated_workspace.state.style.bonds.pair_rules
        )
        before_check = {
            path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
            for path in (manifest_path, workspace_path, output_path)
        }

        checked = subprocess.run(
            command + ["--check"],
            cwd=project_root,
            capture_output=True,
            text=True,
        )

        assert checked.returncode == 0, checked.stderr
        current_source_stat = input_path.stat()
        assert (
            hashlib.sha256(input_path.read_bytes()).hexdigest(),
            current_source_stat.st_size,
            current_source_stat.st_mtime_ns,
            current_source_stat.st_ino,
        ) == source_identity
        assert {
            path: (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
            for path in (manifest_path, workspace_path, output_path)
        } == before_check


def _batch_visualization_preset():
    preset = load_default_style()
    return replace(
        preset,
        style=replace(
            preset.style,
            cell_periodic=replace(
                preset.style.cell_periodic,
                show_unit_cell=0,
            ),
            bonds=BondModuleSettings(
                draw_bonds=True,
                defaults=preset.style.bonds.defaults,
                pair_rules=(
                    BondPairRule("H", "O", 0.5, 1.2, enabled=False),
                    BondPairRule("C", "O", 0.8, 1.6, enabled=True),
                ),
                style=BondStrokeStyle(0.25, "#231815"),
            ),
            export=AppliedExportSettings("svg", 200, True),
        ),
    )

def _svg_bond_groups(path):
    root = ET.parse(path).getroot()
    return [
        node
        for node in root.iter("{http://www.w3.org/2000/svg}g")
        if node.attrib.get("id", "").startswith("bond_")
    ]


def test_batch_preset_is_authoritative_and_preserves_grouped_svg_contract():
    """批处理应逐构型应用同一预设规则，并仍输出单键六对象分组。"""
    input_dir = tempfile.mkdtemp(prefix="meia_preset_input_")
    output_dir = tempfile.mkdtemp(prefix="meia_preset_output_")
    try:
        from ase.io import write

        write(
            os.path.join(input_dir, "disabled_ho.xyz"),
            Atoms("HO", positions=[[0, 0, 0], [0.9, 0, 0]]),
        )
        write(
            os.path.join(input_dir, "enabled_co.xyz"),
            Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]]),
        )

        results = batch_process(
            input_dir,
            output_dir,
            output_format="png",
            visualization_preset=_batch_visualization_preset(),
        )

        assert [os.path.basename(path) for path in results] == [
            "disabled_ho.svg",
            "enabled_co.svg",
        ]
        assert len(_svg_bond_groups(results[0])) == 0
        groups = _svg_bond_groups(results[1])
        assert len(groups) == 1
        assert [child.attrib["data-role"] for child in groups[0]] == [
            "cap-a",
            "cap-b",
            "rect-a",
            "rect-b",
            "outline-1",
            "outline-2",
        ]
    finally:
        shutil.rmtree(input_dir, ignore_errors=True)
        shutil.rmtree(output_dir, ignore_errors=True)


def test_batch_cli_rejects_malformed_preset_before_processing(monkeypatch, tmp_path):
    """损坏的完整预设应在扫描和渲染文件前以非零状态退出。"""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    preset_path = tmp_path / "broken.json"
    preset_path.write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "meia.batch",
            str(input_dir),
            "-o",
            str(output_dir),
            "--preset",
            str(preset_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        batch_main()

    assert exc_info.value.code != 0
    assert not output_dir.exists()


def test_batch_rejects_workspace_snapshot_before_creating_output(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    state = VisualizationState(load_default_style().style)
    snapshot = build_workspace_snapshot(
        Atoms("CO", positions=[[0, 0, 0], [1.2, 0, 0]]),
        "CONTCAR",
        state,
        "saved-work",
        "0.6.0",
    )

    with pytest.raises(PresetError, match="工作状态快照"):
        batch_process(
            str(input_dir),
            str(output_dir),
            visualization_preset=snapshot,
        )

    assert not output_dir.exists()
