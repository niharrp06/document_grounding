from docuxray.geometry import merge_boxes, normalize_box, polygon_to_box
from docuxray.types import BoundingBox


def test_polygon_to_box_uses_extents() -> None:
    box = polygon_to_box([[10, 8], [31, 7], [33, 19], [9, 20]])
    assert box == BoundingBox(9, 7, 33, 20)


def test_merge_boxes_returns_outer_box() -> None:
    box = merge_boxes([BoundingBox(2, 3, 8, 9), BoundingBox(7, 1, 12, 10)])
    assert box == BoundingBox(2, 1, 12, 10)


def test_normalize_box_clamps_to_layoutlm_range() -> None:
    assert normalize_box(BoundingBox(-5, 5, 120, 70), 100, 50) == [0, 100, 1000, 1000]
