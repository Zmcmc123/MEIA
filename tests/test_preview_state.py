"""2D 预览指纹、缓存状态与自动刷新策略。"""

from dataclasses import replace

import numpy as np

from meia.atom_styles import AtomColorStrength, AtomSelectionSettings
from meia.display_complexity import DisplayComplexity
from meia.presets import visual_state_fingerprint
from meia.preview_state import (
    PreviewArtifact,
    PreviewKey,
    PreviewStatus,
    preview_status,
    should_render_preview,
)
from meia.visual_state import VisualizationState, replace_atom_selection


def test_preview_key_changes_for_style_or_camera_but_not_object_identity():
    state = VisualizationState()
    same_value = VisualizationState()
    camera = np.eye(3)

    assert PreviewKey.build("structure-a", state, camera) == PreviewKey.build(
        "structure-a", same_value, camera.copy()
    )

    changed = replace_atom_selection(
        state,
        AtomSelectionSettings(
            color_strengths=(AtomColorStrength(0, "H", 0.3),)
        ),
    )
    assert PreviewKey.build("structure-a", changed, camera) != PreviewKey.build(
        "structure-a", state, camera
    )

    changed_camera = camera.copy()
    changed_camera[0, 0] = 0.5
    assert PreviewKey.build("structure-a", state, changed_camera) != PreviewKey.build(
        "structure-a", state, camera
    )


def test_visual_state_fingerprint_is_deterministic_without_changing_preset_encoding():
    state = VisualizationState()
    before = visual_state_fingerprint(state)
    after = visual_state_fingerprint(replace(state))

    assert before == after
    assert len(before) == 64


def test_preview_cache_distinguishes_missing_current_and_stale():
    current = PreviewKey("structure-a", "state-a", "camera-a")
    old = PreviewKey("structure-a", "state-old", "camera-a")
    artifact = PreviewArtifact(old, b"png", "svg", b"svg")

    assert preview_status(None, current) is PreviewStatus.MISSING
    assert preview_status(artifact, current) is PreviewStatus.STALE
    assert (
        preview_status(replace(artifact, key=current), current)
        is PreviewStatus.CURRENT
    )


def test_small_preview_refreshes_automatically_but_large_preview_requires_click():
    small = DisplayComplexity.from_counts(10, 10, 5, 0)
    large = DisplayComplexity.from_counts(900, 900, 650, 200)

    assert should_render_preview(small, PreviewStatus.MISSING) is True
    assert should_render_preview(small, PreviewStatus.STALE) is True
    assert should_render_preview(small, PreviewStatus.CURRENT) is False
    assert should_render_preview(large, PreviewStatus.MISSING) is False
    assert should_render_preview(large, PreviewStatus.STALE) is False
    assert (
        should_render_preview(
            large,
            PreviewStatus.STALE,
            refresh_requested=True,
        )
        is True
    )
