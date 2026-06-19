from __future__ import annotations

import json
from pathlib import Path

from .dataset import PreparedDocumentDataset, build_label_maps
from .io import read_jsonl


def train_model(
    prepared_manifest: str | Path,
    output_dir: str | Path,
    *,
    base_model: str = "microsoft/layoutlmv3-base",
    epochs: float = 3.0,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
) -> None:
    try:
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are not installed. Install them with: pip install -e '.[ml]'"
        ) from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(prepared_manifest)
    if not records:
        raise ValueError("Prepared manifest contains no records")
    label2id, id2label = build_label_maps(records)
    processor = LayoutLMv3Processor.from_pretrained(base_model, apply_ocr=False)
    dataset = PreparedDocumentDataset(records, processor, label2id)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        base_model,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )
    arguments = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_strategy="epoch",
        logging_steps=10,
        report_to=[],
    )
    trainer = Trainer(model=model, args=arguments, train_dataset=dataset)
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    (output_dir / "labels.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, indent=2) + "\n",
        encoding="utf-8",
    )
