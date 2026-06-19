# DocuXray

DocuXray is a document grounding codebase. It locates the exact occurrence of a requested key-value field in a document and returns the page, bounding box, row information, confidence, and evidence used to make the decision.

The important design choice is that DocuXray does not simply search for a value. It searches for a **field instance**:

```text
key + value + layout relationship + table row/column context + neighboring evidence
```

That makes repeated keys and repeated values manageable in enterprise documents.

## Architecture

```text
Document image/PDF page
  -> OCR words from PaddleOCR or OCR JSON
  -> OCR words with pixel boxes
  -> Line construction
  -> Optional table structure from Table Transformer or another table parser
  -> Document Graph
  -> Field request normalization
  -> Row-aware candidate generation
  -> Optional LayoutLMv3 candidate reranking
  -> Exact bbox selection
  -> Validation-friendly grounded result
```

Current implementation includes:

- PaddleOCR adapter for Python versions supported by PaddlePaddle.
- OCR JSON adapter for Python 3.14 and offline OCR output.
- Document graph data model.
- Field request normalization that preserves duplicate keys.
- Table, row, and cell schema.
- Row-aware table candidate generation.
- Same-line form candidate generation.
- Value-only fallback candidate generation.
- Grounding engine with auditable evidence.
- A `LayoutLMv3Reranker` interface where a trained pairwise reranker can be plugged in.

## Project Layout

```text
src/docuxray/types.py          shared dataclasses and output contracts
src/docuxray/normalization.py  field request and value normalization
src/docuxray/ocr.py            PaddleOCR extraction
src/docuxray/graph.py          document graph construction
src/docuxray/tables.py         table structure loading and graph attachment
src/docuxray/candidates.py     relationship-based candidate generation
src/docuxray/reranking.py      reranker interfaces, including LayoutLMv3 placeholder
src/docuxray/grounding.py      final candidate selection and result construction
src/docuxray/pipeline.py       end-to-end graph-based grounding pipeline
src/docuxray/prepare.py        legacy OCR/BIO preparation flow
src/docuxray/train.py          legacy LayoutLMv3 token-classification training flow
src/docuxray/predict.py        legacy field prediction flow
src/docuxray/cli.py            command-line interface
```

## Setup

Python 3.10 or newer is supported by the core codebase. On Python 3.14, use OCR JSON input because PaddlePaddle and many transformer wheels may not be available for 3.14 yet.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For Python 3.14, this is enough to run the graph-based pipeline with `--ocr-json` or table structure:

```powershell
pip install -e ".[dev]"
```

For live PaddleOCR on Python versions supported by PaddlePaddle:

```powershell
pip install -e ".[ocr]"
```

For transformer training or inference on Python versions supported by PyTorch/Transformers:

```powershell
pip install -e ".[ml]"
```

Or install everything:

```powershell
pip install -e ".[all]"
```

The `ocr` and `ml` extras use Python-version markers. On Python 3.14, unavailable PaddlePaddle, PyTorch, and Transformers packages are skipped so the core package remains installable.

## Field Request Format

Do not use duplicate keys directly in a JSON object. Most JSON parsers keep only the last duplicate key.

Use this instead:

```json
{
  "fields": [
    {
      "field_id": "amount_0",
      "key": "Amount",
      "value": "7000",
      "row_index": 0
    },
    {
      "field_id": "amount_1",
      "key": "Amount",
      "value": "5000",
      "row_index": 1
    }
  ]
}
```

`row_index = 0` means the first data row, `row_index = 1` means the second data row, and so on. Header rows are not counted as data rows.

You can also group related fields as records:

```json
{
  "records": [
    {
      "record_id": "line_0",
      "row_index": 0,
      "fields": {
        "Description": "Consulting",
        "Amount": "7000"
      }
    }
  ]
}
```

## Optional Table Structure

DocuXray is designed to receive table structure from Table Transformer or another parser. The table parser should produce cells with row and column indexes.

Example:

```json
{
  "tables": [
    {
      "table_id": "table_0",
      "page": 0,
      "headers": ["Description", "Amount"],
      "rows": [
        {
          "row_index": 0,
          "cells": [
            {"text": "Consulting", "box": [100, 180, 260, 210]},
            {"text": "7000", "box": [300, 180, 380, 210]}
          ]
        },
        {
          "row_index": 1,
          "cells": [
            {"text": "Support", "box": [100, 220, 260, 250]},
            {"text": "5000", "box": [300, 220, 380, 250]}
          ]
        }
      ]
    }
  ]
}
```

For production, replace the example JSON table loader with a Table Transformer adapter that emits the same `DocumentTable`, `TableRow`, and `TableCell` objects.

## Optional OCR JSON

On Python 3.14, provide OCR words as JSON instead of running PaddleOCR directly:

```json
{
  "words": [
    {
      "id": "w0",
      "text": "Amount",
      "box": [300, 140, 380, 170],
      "confidence": 0.99,
      "page": 0
    },
    {
      "id": "w1",
      "text": "5000",
      "box": [300, 220, 380, 250],
      "confidence": 0.99,
      "page": 0
    }
  ]
}
```

This lets you run DocuXray with OCR from any source: PaddleOCR in another environment, Azure OCR, Textract, Tesseract, Google Vision, or manually prepared test data.

## Run Row-Aware Grounding

Create a field request file:

```powershell
notepad examples\row_aware_fields.json
```

Run grounding:

```powershell
docuxray ground `
  --image path\to\document.png `
  --fields examples\row_aware_fields.json `
  --tables examples\table_structure.json `
  --ocr-json examples\ocr_words.json `
  --output-graph outputs\document_graph.json
```

If you only want to test table-aware grounding and do not need OCR words, use:

```powershell
docuxray ground `
  --image path\to\document.png `
  --fields examples\row_aware_fields.json `
  --tables examples\table_structure.json `
  --ocr-backend none `
  --output-graph outputs\document_graph.json
```

Output:

```json
[
  {
    "field_id": "amount_row_1",
    "key": "Amount",
    "value": "5000",
    "matched_text": "5000",
    "page": 0,
    "bbox": [300, 220, 380, 250],
    "confidence": 1.0,
    "status": "matched",
    "source": "TABLE_HEADER_TO_CELL",
    "row_index": 1,
    "col_index": 1,
    "table_id": "table_0",
    "evidence": {
      "value_match": 1.0,
      "header_match": 1.0,
      "row_index_match": true
    }
  }
]
```

## Legacy Training Flow

The original token-classification path is still available.

Create a JSONL manifest:

```json
{"image":"images/invoice-001.png","fields":{"invoice_number":"INV-001","customer_name":"Ada Lovelace"}}
```

Prepare OCR and BIO labels:

```powershell
docuxray prepare --manifest examples/train_manifest.jsonl --output outputs/train.prepared.jsonl
```

This legacy command requires live PaddleOCR, so it is not the recommended path on Python 3.14.

Fine-tune LayoutLMv3:

```powershell
docuxray train --prepared-manifest outputs/train.prepared.jsonl --output-dir outputs/layoutlmv3-docuxray
```

Run the legacy predictor:

```powershell
docuxray predict --image images/invoice-001.png --fields examples/requested_fields.json --model-dir outputs/layoutlmv3-docuxray
```

## Where To Add Table Transformer

Add a new adapter under `src/docuxray/tables.py` or a new module such as `src/docuxray/table_transformer.py`.

The adapter should:

1. Detect table boundaries.
2. Detect rows, columns, headers, and cells.
3. Assign OCR words to cells.
4. Emit `DocumentTable`, `TableRow`, and `TableCell` objects.
5. Preserve `row_index`, `col_index`, `header_text`, `bbox`, and confidence.

The grounding engine does not need to know which model produced the table structure.

## Where To Add LayoutLMv3 Reranking

`src/docuxray/reranking.py` contains:

```python
class CandidateReranker:
    def score(self, request, candidates, graph):
        ...
```

Implement `LayoutLMv3Reranker.score()` after you have training data for candidate pairs. The reranker should receive top-K candidates from rule-based generation and return the same candidates with improved scores.

This keeps LayoutLMv3 focused on the hard problem: choosing the correct key-value relationship among plausible candidates.

## Testing

Install dev dependencies:

```powershell
pip install -e ".[dev]"
```

Run tests:

```powershell
python -m pytest
```

If `pytest` is not installed, at least run:

```powershell
python -m compileall src tests
```

## Production Notes

Keep these rules as the codebase grows:

- Preserve raw OCR text and normalized text separately.
- Keep all bboxes in original page pixel coordinates unless explicitly converted.
- Treat tables, rows, columns, and cells as first-class objects.
- Do candidate generation before transformer reranking.
- Return evidence with every prediction.
- Use `ambiguous`, `low_confidence`, and `not_found` statuses instead of forcing a bad match.
- Evaluate strict grounding accuracy: key, value, row, page, and bbox must all be correct.
