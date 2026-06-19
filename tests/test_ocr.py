from docuxray.ocr import _iter_lines


def test_iter_lines_flattens_paddleocr_pages() -> None:
    line = [[[1, 2], [3, 2], [3, 4], [1, 4]], ("INV-001", 0.98)]
    assert list(_iter_lines([[line]])) == [line]
