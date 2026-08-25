"""Streamlit 会话内 2D 预览产物的生命周期。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json

import numpy as np

from .display_complexity import DisplayComplexity
from .presets import visual_state_fingerprint
from .visual_state import VisualizationState


class PreviewStatus(str, Enum):
    MISSING = "missing"
    CURRENT = "current"
    STALE = "stale"


@dataclass(frozen=True)
class PreviewKey:
    structure_id: str
    visual_state_sha256: str
    camera_sha256: str

    @classmethod
    def build(
        cls,
        structure_id: str,
        state: VisualizationState,
        rotation_matrix: np.ndarray,
    ) -> "PreviewKey":
        if not isinstance(structure_id, str) or not structure_id:
            raise ValueError("structure_id must be a non-empty string")
        matrix = np.asarray(rotation_matrix, dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError("rotation_matrix must be a finite 3x3 matrix")
        camera_payload = json.dumps(
            matrix.tolist(),
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return cls(
            structure_id=structure_id,
            visual_state_sha256=visual_state_fingerprint(state),
            camera_sha256=sha256(camera_payload).hexdigest(),
        )


@dataclass(frozen=True)
class PreviewArtifact:
    key: PreviewKey
    preview_png: bytes
    export_format: str
    export_bytes: bytes


def preview_status(
    artifact: PreviewArtifact | None,
    current_key: PreviewKey,
) -> PreviewStatus:
    if artifact is None:
        return PreviewStatus.MISSING
    if artifact.key == current_key:
        return PreviewStatus.CURRENT
    return PreviewStatus.STALE


def should_render_preview(
    complexity: DisplayComplexity,
    status: PreviewStatus,
    *,
    refresh_requested: bool = False,
) -> bool:
    if not isinstance(complexity, DisplayComplexity):
        raise TypeError("complexity must be DisplayComplexity")
    status = PreviewStatus(status)
    if complexity.manual_2d_recommended:
        return bool(refresh_requested)
    return status is not PreviewStatus.CURRENT
