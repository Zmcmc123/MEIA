"""大体系侧栏原子选择的纯分页模型。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


LARGE_SELECTION_THRESHOLD = 1_000
ATOM_SELECTION_PAGE_SIZE = 200


def _non_negative_indices(values: Iterable[int], label: str) -> tuple[int, ...]:
    indices = tuple(values)
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in indices
    ):
        raise ValueError(f"{label} must contain non-negative integer indices")
    return tuple(sorted(set(indices)))


@dataclass(frozen=True)
class AtomSelectionPage:
    atom_count: int
    page_number: int
    page_size: int
    page_count: int
    indices: tuple[int, ...]


def selection_page(
    atom_count: int,
    page_number: int,
    page_size: int = ATOM_SELECTION_PAGE_SIZE,
) -> AtomSelectionPage:
    if (
        isinstance(atom_count, bool)
        or not isinstance(atom_count, int)
        or atom_count < 0
    ):
        raise ValueError("atom_count must be a non-negative integer")
    if (
        isinstance(page_size, bool)
        or not isinstance(page_size, int)
        or page_size <= 0
    ):
        raise ValueError("page_size must be a positive integer")
    if isinstance(page_number, bool) or not isinstance(page_number, int):
        raise ValueError("page_number must be an integer")
    page_count = (atom_count + page_size - 1) // page_size
    if not 1 <= page_number <= page_count:
        raise ValueError(
            f"page_number must be between 1 and {page_count}"
        )
    start = (page_number - 1) * page_size
    end = min(start + page_size, atom_count)
    return AtomSelectionPage(
        atom_count=atom_count,
        page_number=page_number,
        page_size=page_size,
        page_count=page_count,
        indices=tuple(range(start, end)),
    )


def apply_page_selection(
    current_indices: Iterable[int],
    page_selected_indices: Iterable[int],
    action: str,
    *,
    allowed_indices: Iterable[int] | None = None,
) -> tuple[int, ...]:
    current = set(_non_negative_indices(current_indices, "current_indices"))
    page_selected = set(
        _non_negative_indices(page_selected_indices, "page_selected_indices")
    )
    if action not in {"add", "remove"}:
        raise ValueError("page selection action must be 'add' or 'remove'")
    if allowed_indices is not None:
        allowed = set(_non_negative_indices(allowed_indices, "allowed_indices"))
        if not page_selected <= allowed:
            raise ValueError("page selection contains an index outside the active page")
    if action == "add":
        current.update(page_selected)
    else:
        current.difference_update(page_selected)
    return tuple(sorted(current))
