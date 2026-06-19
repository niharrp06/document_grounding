from __future__ import annotations

from collections.abc import Iterable, Sequence

from .types import BoundingBox


def polygon_to_box(points: Sequence[Sequence[float]]) -> BoundingBox:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return BoundingBox(round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)))


def merge_boxes(boxes: Iterable[BoundingBox]) -> BoundingBox | None:
    boxes = list(boxes)
    if not boxes:
        return None
    return BoundingBox(
        min(box.x1 for box in boxes),
        min(box.y1 for box in boxes),
        max(box.x2 for box in boxes),
        max(box.y2 for box in boxes),
    )


def normalize_box(box: BoundingBox, width: int, height: int) -> list[int]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive")
    return [
        max(0, min(1000, round(1000 * box.x1 / width))),
        max(0, min(1000, round(1000 * box.y1 / height))),
        max(0, min(1000, round(1000 * box.x2 / width))),
        max(0, min(1000, round(1000 * box.y2 / height))),
    ]
