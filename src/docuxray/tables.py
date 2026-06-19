from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .types import BoundingBox, DocumentTable, TableCell, TableRow


def load_tables(path: str | Path | None) -> tuple[list[DocumentTable], list[TableRow], list[TableCell]]:
    if path is None:
        return [], [], []
    table_path = Path(path)
    if table_path.suffix.casefold() == ".json":
        return _load_json_tables(table_path)
    if table_path.suffix.casefold() == ".csv":
        return _load_csv_table(table_path)
    raise ValueError("Table structure must be provided as .json or .csv")


def _load_json_tables(path: Path) -> tuple[list[DocumentTable], list[TableRow], list[TableCell]]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    tables: list[DocumentTable] = []
    rows: list[TableRow] = []
    cells: list[TableCell] = []
    for table_index, table in enumerate(data.get("tables", [])):
        table_id = str(table.get("table_id") or f"table_{table_index}")
        page = int(table.get("page", 0))
        table_cell_ids: list[str] = []
        row_ids: list[str] = []
        headers = [str(header) for header in table.get("headers", [])]
        for row_index, row in enumerate(table.get("rows", [])):
            is_header = bool(row.get("is_header", False))
            logical_row_index = -1 if is_header else int(row.get("row_index", row_index))
            row_cell_ids: list[str] = []
            for col_index, cell in enumerate(row.get("cells", [])):
                cell_id = str(cell.get("cell_id") or f"{table_id}_r{logical_row_index}_c{col_index}")
                cell_box = BoundingBox(*cell["box"])
                header_text = cell.get("header_text")
                if header_text is None and 0 <= col_index < len(headers):
                    header_text = headers[col_index]
                cells.append(
                    TableCell(
                        cell_id=cell_id,
                        page=page,
                        table_id=table_id,
                        row_index=logical_row_index,
                        col_index=col_index,
                        text=str(cell.get("text", "")),
                        box=cell_box,
                        word_ids=list(cell.get("word_ids", [])),
                        is_header=is_header,
                        header_text=header_text,
                        confidence=float(cell.get("confidence", 1.0)),
                    )
                )
                row_cell_ids.append(cell_id)
                table_cell_ids.append(cell_id)
            if row_cell_ids:
                row_box = _box_from_cells(cells[-len(row_cell_ids) :])
                row_id = f"{table_id}_row_{logical_row_index}"
                rows.append(TableRow(row_id, page, table_id, logical_row_index, row_cell_ids, row_box, is_header))
                row_ids.append(row_id)
        table_box = BoundingBox(*table["box"]) if "box" in table else _box_from_cells([c for c in cells if c.table_id == table_id])
        tables.append(DocumentTable(table_id, page, table_box, row_ids, table_cell_ids, float(table.get("confidence", 1.0))))
    return tables, rows, cells


def _load_csv_table(path: Path) -> tuple[list[DocumentTable], list[TableRow], list[TableCell]]:
    rows_data = list(csv.reader(path.open(encoding="utf-8")))
    if not rows_data:
        return [], [], []
    headers = rows_data[0]
    cells: list[TableCell] = []
    rows: list[TableRow] = []
    cell_width = 120
    cell_height = 32
    table_id = "csv_table_0"
    for row_index, row_data in enumerate(rows_data[1:]):
        row_cell_ids = []
        for col_index, text in enumerate(row_data):
            cell_id = f"{table_id}_r{row_index}_c{col_index}"
            box = BoundingBox(col_index * cell_width, (row_index + 1) * cell_height, (col_index + 1) * cell_width, (row_index + 2) * cell_height)
            cells.append(TableCell(cell_id, 0, table_id, row_index, col_index, text, box, [], False, headers[col_index] if col_index < len(headers) else None))
            row_cell_ids.append(cell_id)
        rows.append(TableRow(f"{table_id}_row_{row_index}", 0, table_id, row_index, row_cell_ids, _box_from_cells(cells[-len(row_cell_ids) :])))
    table = DocumentTable(table_id, 0, _box_from_cells(cells), [row.row_id for row in rows], [cell.cell_id for cell in cells])
    return [table], rows, cells


def attach_tables(graph: Any, tables: list[DocumentTable], rows: list[TableRow], cells: list[TableCell]) -> Any:
    graph.tables = tables
    graph.rows = rows
    graph.cells = cells
    return graph


def _box_from_cells(cells: list[TableCell]) -> BoundingBox:
    return BoundingBox(
        min(cell.box.x1 for cell in cells),
        min(cell.box.y1 for cell in cells),
        max(cell.box.x2 for cell in cells),
        max(cell.box.y2 for cell in cells),
    )
