from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int

    def __post_init__(self) -> None:
        if self.x1 > self.x2 or self.y1 > self.y2:
            raise ValueError(f"Invalid bounding box: {self}")

    def to_list(self) -> list[int]:
        return [self.x1, self.y1, self.x2, self.y2]


@dataclass(frozen=True)
class OCRWord:
    text: str
    box: BoundingBox
    confidence: float
    page: int = 0
    word_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.word_id,
            "text": self.text,
            "box": self.box.to_list(),
            "confidence": self.confidence,
            "page": self.page,
        }


@dataclass(frozen=True)
class FieldBox:
    field: str
    value: str
    box: BoundingBox | None
    confidence: float
    source: str

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = self.box.to_list() if self.box else None
        return result


@dataclass(frozen=True)
class FieldRequest:
    field_id: str
    key: str
    value: str
    row_index: int | None = None
    record_id: str | None = None
    value_type: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentPage:
    page: int
    width: int
    height: int
    image_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DocumentLine:
    line_id: str
    page: int
    text: str
    word_ids: list[str]
    box: BoundingBox

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = self.box.to_list()
        return result


@dataclass(frozen=True)
class TableCell:
    cell_id: str
    page: int
    table_id: str
    row_index: int
    col_index: int
    text: str
    box: BoundingBox
    word_ids: list[str]
    is_header: bool = False
    header_text: str | None = None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = self.box.to_list()
        return result


@dataclass(frozen=True)
class TableRow:
    row_id: str
    page: int
    table_id: str
    row_index: int
    cell_ids: list[str]
    box: BoundingBox
    is_header: bool = False

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = self.box.to_list()
        return result


@dataclass(frozen=True)
class DocumentTable:
    table_id: str
    page: int
    box: BoundingBox
    row_ids: list[str]
    cell_ids: list[str]
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["box"] = self.box.to_list()
        return result


class CandidateRelationship(str, Enum):
    TABLE_HEADER_TO_CELL = "TABLE_HEADER_TO_CELL"
    SAME_LINE = "SAME_LINE"
    VALUE_ONLY = "VALUE_ONLY"


@dataclass(frozen=True)
class GroundingCandidate:
    candidate_id: str
    field_id: str
    key: str
    value: str
    matched_text: str
    page: int
    value_box: BoundingBox
    confidence: float
    relationship: CandidateRelationship
    key_box: BoundingBox | None = None
    table_id: str | None = None
    row_index: int | None = None
    col_index: int | None = None
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["relationship"] = self.relationship.value
        result["value_box"] = self.value_box.to_list()
        result["key_box"] = self.key_box.to_list() if self.key_box else None
        return result


@dataclass(frozen=True)
class GroundedField:
    field_id: str
    key: str
    value: str
    matched_text: str | None
    page: int | None
    bbox: BoundingBox | None
    confidence: float
    status: str
    source: str
    row_index: int | None = None
    col_index: int | None = None
    table_id: str | None = None
    evidence: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["bbox"] = self.bbox.to_list() if self.bbox else None
        return result
