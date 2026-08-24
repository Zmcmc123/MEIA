"""基于显示球半径裁剪三维化学键中心线。"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BondSurfaceSegment:
    """两个球面交点之间仍暴露在原子球外的键段。"""

    start: np.ndarray
    midpoint: np.ndarray
    end: np.ndarray


def clip_bond_to_spheres(
    center_a: np.ndarray,
    center_b: np.ndarray,
    radius_a: float,
    radius_b: float,
    *,
    tolerance: float = 1e-12,
) -> BondSurfaceSegment | None:
    """返回中心连线与两个显示球面的交点；无外露键段时返回 ``None``。"""
    point_a = np.asarray(center_a, dtype=float)
    point_b = np.asarray(center_b, dtype=float)
    delta = point_b - point_a
    distance = float(np.linalg.norm(delta))
    if distance <= radius_a + radius_b + tolerance:
        return None

    direction = delta / distance
    start = point_a + direction * radius_a
    end = point_b - direction * radius_b
    return BondSurfaceSegment(
        start=start,
        midpoint=(start + end) / 2,
        end=end,
    )
