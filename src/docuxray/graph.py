from __future__ import annotations

from dataclasses import dataclass, field

from .geometry import merge_boxes
from .types import (
    DocumentLine,
    DocumentPage,
    DocumentTable,
    OCRWord,
    TableCell,
    TableRow,
)


@dataclass
class DocumentGraph:
    pages: list[DocumentPage]
    words: list[OCRWord]
    lines: list[DocumentLine] = field(default_factory=list)
    tables: list[DocumentTable] = field(default_factory=list)
    rows: list[TableRow] = field(default_factory=list)
    cells: list[TableCell] = field(default_factory=list)

    def words_by_id(self) -> dict[str, OCRWord]:
        return {word.word_id or f"p{word.page}_w{index}": word for index, word in enumerate(self.words)}

    def cells_by_id(self) -> dict[str, TableCell]:
        return {cell.cell_id: cell for cell in self.cells}

    def table_cells(self, table_id: str) -> list[TableCell]:
        return [cell for cell in self.cells if cell.table_id == table_id]

    def to_dict(self) -> dict[str, object]:
        return {
            "pages": [page.to_dict() for page in self.pages],
            "words": [word.to_dict() for word in self.words],
            "lines": [line.to_dict() for line in self.lines],
            "tables": [table.to_dict() for table in self.tables],
            "rows": [row.to_dict() for row in self.rows],
            "cells": [cell.to_dict() for cell in self.cells],
        }


def build_document_graph(
    pages: list[DocumentPage],
    words: list[OCRWord],
    *,
    line_y_tolerance: int = 8,
) -> DocumentGraph:
    numbered_words = [
        word if word.word_id else OCRWord(word.text, word.box, word.confidence, word.page, f"p{word.page}_w{index}")
        for index, word in enumerate(words)
    ]
    return DocumentGraph(pages=pages, words=numbered_words, lines=_build_lines(numbered_words, line_y_tolerance))


def _build_lines(words: list[OCRWord], y_tolerance: int) -> list[DocumentLine]:
    lines: list[list[OCRWord]] = []
    for word in sorted(words, key=lambda item: (item.page, item.box.y1, item.box.x1)):
        placed = False
        center_y = (word.box.y1 + word.box.y2) / 2
        for line_words in lines:
            first = line_words[0]
            if first.page != word.page:
                continue
            first_center_y = (first.box.y1 + first.box.y2) / 2
            if abs(center_y - first_center_y) <= y_tolerance:
                line_words.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])

    document_lines: list[DocumentLine] = []
    for index, line_words in enumerate(lines):
        ordered = sorted(line_words, key=lambda item: item.box.x1)
        box = merge_boxes(word.box for word in ordered)
        if box is None:
            continue
        page = ordered[0].page
        document_lines.append(
            DocumentLine(
                line_id=f"p{page}_l{index}",
                page=page,
                text=" ".join(word.text for word in ordered),
                word_ids=[word.word_id or "" for word in ordered],
                box=box,
            )
        )
    return document_lines
