from __future__ import annotations

from pathlib import Path
from typing import Any

from .geometry import normalize_box
from .io import ocr_words_from_dicts, read_jsonl


def build_label_maps(records: list[dict[str, Any]]) -> tuple[dict[str, int], dict[int, str]]:
    labels = {"O"}
    for record in records:
        labels.update(record["labels"])
    ordered = ["O", *sorted(label for label in labels if label != "O")]
    label2id = {label: index for index, label in enumerate(ordered)}
    return label2id, {index: label for label, index in label2id.items()}


class PreparedDocumentDataset:
    def __init__(self, records: list[dict[str, Any]], processor: Any, label2id: dict[str, int]):
        self.records = records
        self.processor = processor
        self.label2id = label2id

    @classmethod
    def from_jsonl(
        cls, path: str | Path, processor: Any, label2id: dict[str, int]
    ) -> "PreparedDocumentDataset":
        return cls(read_jsonl(path), processor, label2id)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Pillow is not installed. Install dependencies with: pip install -e ."
            ) from exc

        record = self.records[index]
        words = ocr_words_from_dicts(record["words"])
        boxes = [normalize_box(word.box, record["width"], record["height"]) for word in words]
        labels = [self.label2id[label] for label in record["labels"]]
        with Image.open(record["image"]) as image:
            encoding = self.processor(
                image.convert("RGB"),
                [word.text for word in words],
                boxes=boxes,
                word_labels=labels,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
        return {key: value.squeeze(0) for key, value in encoding.items()}
