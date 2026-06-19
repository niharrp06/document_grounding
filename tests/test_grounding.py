from docuxray.graph import DocumentGraph
from docuxray.grounding import GroundingEngine
from docuxray.types import BoundingBox, DocumentPage, DocumentTable, FieldRequest, TableCell, TableRow


def test_grounding_prefers_requested_table_row() -> None:
    cells = [
        TableCell("c0", 0, "t0", 0, 0, "7000", BoundingBox(10, 10, 40, 30), [], False, "Amount"),
        TableCell("c1", 0, "t0", 1, 0, "5000", BoundingBox(10, 40, 40, 60), [], False, "Amount"),
    ]
    graph = DocumentGraph(
        pages=[DocumentPage(0, 100, 100)],
        words=[],
        tables=[DocumentTable("t0", 0, BoundingBox(0, 0, 100, 80), ["r0", "r1"], ["c0", "c1"])],
        rows=[
            TableRow("r0", 0, "t0", 0, ["c0"], BoundingBox(0, 10, 100, 30)),
            TableRow("r1", 0, "t0", 1, ["c1"], BoundingBox(0, 40, 100, 60)),
        ],
        cells=cells,
    )
    result = GroundingEngine().ground(graph, [FieldRequest("amount_1", "Amount", "5000", row_index=1)])[0]
    assert result.status == "matched"
    assert result.row_index == 1
    assert result.bbox == BoundingBox(10, 40, 40, 60)
