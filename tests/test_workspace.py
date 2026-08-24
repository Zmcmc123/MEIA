"""上传构型、通用风格和工作状态快照的会话转换。"""

from dataclasses import replace
import json

import numpy as np
import pytest
from ase import Atoms

from meia.presets import (
    SCHEMA_VERSION,
    PresetError,
    PresetKind,
    WorkspaceSnapshot,
    style_preset_to_json,
    workspace_snapshot_to_json,
    load_default_style,
)
from meia.locale_state import (
    APP_LOCALE_KEY,
    APP_LOCALE_SOURCE_KEY,
    APP_LOCALE_WIDGET_KEY,
)
from meia.i18n import I18n, Locale
from meia.periodic_display import CellPeriodicSettings, PeriodicRange
from meia.visual_state import VisualizationState
from meia.workspace import (
    ActiveWorkspace,
    activate_snapshot,
    activate_upload,
    build_style_preset,
    build_workspace_snapshot,
    canonical_structure_bytes,
    confirm_pending_snapshot,
    stage_snapshot,
    structure_identity,
)


def _atoms(distance=1.2):
    return Atoms(
        "CO",
        positions=[[0, 0, 0], [distance, 0, 0]],
        cell=[8, 9, 10],
        pbc=[True, True, False],
    )


def test_structure_identity_is_canonical_and_sensitive_to_structure_fields():
    atoms = _atoms()

    assert structure_identity(atoms) == structure_identity(atoms.copy())
    assert structure_identity(atoms) != structure_identity(_atoms(1.3))
    assert structure_identity(atoms) != structure_identity(
        Atoms("CO", positions=atoms.positions, cell=[8, 9, 11], pbc=atoms.pbc)
    )
    assert structure_identity(atoms) != structure_identity(
        Atoms("CO", positions=atoms.positions, cell=atoms.cell, pbc=True)
    )


def test_unchanged_uploader_does_not_replace_a_confirmed_snapshot():
    uploaded, token, replaced = activate_upload(
        None,
        None,
        b"original file",
        "CONTCAR",
        _atoms(),
    )
    assert replaced is True

    snapshot_state = VisualizationState(load_default_style().style)
    snapshot = build_workspace_snapshot(
        _atoms(1.5),
        "saved.legacy",
        snapshot_state,
        "saved-work",
        "0.6.0",
    )
    active_snapshot = activate_snapshot(snapshot)

    unchanged, unchanged_token, replaced = activate_upload(
        active_snapshot,
        token,
        b"original file",
        "CONTCAR",
        _atoms(),
    )

    assert unchanged is active_snapshot
    assert unchanged_token == token
    assert replaced is False


def test_new_upload_replaces_active_workspace_and_copies_atoms():
    active, token, replaced = activate_upload(
        None,
        None,
        b"first",
        "POSCAR",
        _atoms(),
    )
    source = _atoms(1.4)

    updated, new_token, replaced = activate_upload(
        active,
        token,
        b"second",
        "CONTCAR",
        source,
    )
    source.positions[0, 0] = 99

    assert replaced is True
    assert new_token != token
    assert updated.source_name == "CONTCAR"
    assert updated.atoms.positions[0, 0] == 0


def test_snapshot_is_staged_without_mutating_workspace_until_confirmed():
    state = VisualizationState(load_default_style().style)
    snapshot = build_workspace_snapshot(
        _atoms(1.6), "CONTCAR", state, "saved-work", "0.6.0"
    )
    pending = stage_snapshot(snapshot)

    assert pending.source_name == "CONTCAR"
    assert pending.atom_count == 2

    with pytest.raises(ValueError, match="确认"):
        confirm_pending_snapshot(pending, confirmed=False)

    active, restored_state = confirm_pending_snapshot(pending, confirmed=True)
    assert isinstance(active, ActiveWorkspace)
    assert np.allclose(active.atoms.positions[1], [1.6, 0, 0])
    assert restored_state == state


def test_snapshot_confirmation_error_has_exact_english_diagnostic():
    state = VisualizationState(load_default_style().style)
    snapshot = build_workspace_snapshot(
        _atoms(), "CONTCAR", state, "saved-work", "0.11.0"
    )
    with pytest.raises(ValueError) as captured:
        confirm_pending_snapshot(stage_snapshot(snapshot), confirmed=False)

    assert I18n(Locale.EN).error_text(
        captured.value, "file.snapshot_not_applied"
    ) == (
        "Confirm the workspace snapshot before replacing the current structure."
    )


def test_snapshot_confirmation_normalizes_non_periodic_axes_atomically():
    atoms = _atoms(1.6)
    state = VisualizationState(
        replace(
            load_default_style().style,
            cell_periodic=CellPeriodicSettings(
                a=PeriodicRange(-1, 2),
                b=PeriodicRange(0, 3),
                c=PeriodicRange(-50_000, 50_000),
            ),
        )
    )
    snapshot = build_workspace_snapshot(
        atoms, "CONTCAR", state, "saved-work", "0.8.0"
    )

    active, restored_state = confirm_pending_snapshot(
        stage_snapshot(snapshot), confirmed=True
    )

    assert active.atoms == atoms
    assert restored_state.style.cell_periodic.a == PeriodicRange(-1, 2)
    assert restored_state.style.cell_periodic.b == PeriodicRange(0, 3)
    assert restored_state.style.cell_periodic.c == PeriodicRange(0, 1)
    assert snapshot.state.style.cell_periodic.c == PeriodicRange(-50_000, 50_000)


def test_snapshot_confirmation_rejects_periodic_display_over_instance_limit():
    atoms = _atoms()
    state = VisualizationState(
        replace(
            load_default_style().style,
            cell_periodic=CellPeriodicSettings(a=PeriodicRange(0, 25_001)),
        )
    )
    snapshot = build_workspace_snapshot(
        atoms, "CONTCAR", state, "too-large", "0.8.0"
    )

    with pytest.raises(PresetError, match="50,000"):
        confirm_pending_snapshot(stage_snapshot(snapshot), confirmed=True)


def test_two_exports_have_intentionally_different_scope():
    atoms = _atoms()
    state = VisualizationState(load_default_style().style)

    style = build_style_preset(state, "paper-style", "0.6.0")
    snapshot = build_workspace_snapshot(
        atoms, "CONTCAR", state, "saved-work", "0.6.0"
    )

    assert style.metadata.preset_kind is PresetKind.STYLE
    assert style.metadata.schema_version == SCHEMA_VERSION
    assert style.style == state.style
    assert snapshot.metadata.preset_kind is PresetKind.WORKSPACE_SNAPSHOT
    assert snapshot.metadata.schema_version == SCHEMA_VERSION
    assert snapshot.structure.to_atoms() == atoms
    assert snapshot.state is state
    assert canonical_structure_bytes(atoms) == canonical_structure_bytes(
        snapshot.structure.to_atoms()
    )


def test_interface_language_never_changes_or_enters_serialized_state():
    atoms = _atoms()
    state = VisualizationState(load_default_style().style)
    style = build_style_preset(state, "language-invariant", "0.11.0")
    snapshot = build_workspace_snapshot(
        atoms,
        "CONTCAR",
        state,
        "language-invariant",
        "0.11.0",
    )
    session_state = {
        APP_LOCALE_KEY: "zh-CN",
        APP_LOCALE_SOURCE_KEY: "manual",
        APP_LOCALE_WIDGET_KEY: "zh-CN",
    }
    style_before = style_preset_to_json(style)
    snapshot_before = workspace_snapshot_to_json(snapshot)

    session_state.update(
        {
            APP_LOCALE_KEY: "en",
            APP_LOCALE_SOURCE_KEY: "manual",
            APP_LOCALE_WIDGET_KEY: "en",
        }
    )
    style_after = style_preset_to_json(style)
    snapshot_after = workspace_snapshot_to_json(snapshot)

    assert style_before == style_after
    assert snapshot_before == snapshot_after
    for payload in (style_after, snapshot_after):
        decoded = json.loads(payload)
        assert decoded["schema_version"] == 7
        assert decoded["meia_version"] == "0.11.0"
        for forbidden in (
            "locale",
            "zh-CN",
            APP_LOCALE_KEY,
            APP_LOCALE_WIDGET_KEY,
        ):
            assert forbidden not in payload
