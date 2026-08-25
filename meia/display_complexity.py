"""根据实际显示实例估算二维与三维交互负载。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .visual_state import RenderContext


MANUAL_2D_ARTIST_THRESHOLD = 5_000
LARGE_3D_ATOM_THRESHOLD = 5_000
EXTREME_3D_ATOM_THRESHOLD = 20_000


def _non_negative_count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name}必须是整数")
    if value < 0:
        raise ValueError(f"{name}不能小于 0")
    return value


@dataclass(frozen=True)
class DisplayComplexity:
    """用于选择交互策略的确定性显示复杂度。"""

    source_atom_count: int
    atom_instance_count: int
    visible_bond_instance_count: int
    hydrogen_bond_instance_count: int

    def __post_init__(self) -> None:
        for name in (
            "source_atom_count",
            "atom_instance_count",
            "visible_bond_instance_count",
            "hydrogen_bond_instance_count",
        ):
            object.__setattr__(
                self,
                name,
                _non_negative_count(getattr(self, name), name),
            )

    @classmethod
    def from_counts(
        cls,
        source_atom_count: int,
        atom_instance_count: int,
        visible_bond_instance_count: int,
        hydrogen_bond_instance_count: int,
    ) -> "DisplayComplexity":
        return cls(
            source_atom_count=source_atom_count,
            atom_instance_count=atom_instance_count,
            visible_bond_instance_count=visible_bond_instance_count,
            hydrogen_bond_instance_count=hydrogen_bond_instance_count,
        )

    @property
    def estimated_2d_artist_count(self) -> int:
        return (
            self.atom_instance_count
            + 6 * self.visible_bond_instance_count
            + self.hydrogen_bond_instance_count
        )

    @property
    def manual_2d_recommended(self) -> bool:
        return self.estimated_2d_artist_count >= MANUAL_2D_ARTIST_THRESHOLD

    @property
    def large_3d_interaction(self) -> bool:
        return self.atom_instance_count >= LARGE_3D_ATOM_THRESHOLD

    @property
    def extreme_3d_interaction(self) -> bool:
        return self.atom_instance_count >= EXTREME_3D_ATOM_THRESHOLD


def measure_display_complexity(
    source_atom_count: int,
    context: "RenderContext",
) -> DisplayComplexity:
    """统计隐藏原子过滤后的完整显示实例。"""
    hidden = frozenset(context.hidden_atom_indices)
    atom_instance_count = sum(
        instance.source_atom_index not in hidden
        for instance in context.periodic_display.atom_instances
    )
    visible_bond_instance_count = sum(
        instance.source_bond.visible
        and instance.source_bond.i not in hidden
        and instance.source_bond.j not in hidden
        for instance in context.periodic_display.bond_instances
    )
    return DisplayComplexity.from_counts(
        source_atom_count,
        atom_instance_count,
        visible_bond_instance_count,
        sum(1 for hydrogen_bond in context.hydrogen_bonds if hydrogen_bond.visible),
    )
