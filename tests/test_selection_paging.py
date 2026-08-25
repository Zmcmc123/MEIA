"""大体系原子选择的纯分页语义。"""

import pytest

from meia.selection_paging import (
    ATOM_SELECTION_PAGE_SIZE,
    LARGE_SELECTION_THRESHOLD,
    apply_page_selection,
    selection_page,
)


def test_page_selection_adds_and_removes_without_touching_other_pages():
    page = selection_page(atom_count=2500, page_number=2, page_size=200)

    assert page.indices[0] == 200
    assert page.indices[-1] == 399
    assert page.page_count == 13
    assert apply_page_selection((5, 205, 900), (210, 211), "add") == (
        5,
        205,
        210,
        211,
        900,
    )
    assert apply_page_selection((5, 205, 210, 900), (205, 210), "remove") == (
        5,
        900,
    )
    assert LARGE_SELECTION_THRESHOLD == 1000
    assert ATOM_SELECTION_PAGE_SIZE == 200


@pytest.mark.parametrize(
    "kwargs",
    (
        {"atom_count": 2500, "page_number": 0, "page_size": 200},
        {"atom_count": 2500, "page_number": 14, "page_size": 200},
        {"atom_count": 2500, "page_number": 1, "page_size": 0},
    ),
)
def test_selection_page_rejects_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        selection_page(**kwargs)


def test_page_action_rejects_an_index_outside_the_active_page():
    page = selection_page(atom_count=2500, page_number=2, page_size=200)

    with pytest.raises(ValueError, match="active page"):
        apply_page_selection(
            (5, 900),
            (199, 210),
            "add",
            allowed_indices=page.indices,
        )
