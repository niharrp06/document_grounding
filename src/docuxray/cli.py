from __future__ import annotations

import argparse
import json

from .io import read_fields
from .pipeline import run_grounding_pipeline
from .predict import predict_fields
from .prepare import prepare_manifest
from .train import train_model
from .annotation_preprocessor import convert_path


def main() -> None:
    parser = argparse.ArgumentParser(prog="docuxray")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Run PaddleOCR and create a training JSONL file")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--lang", default="en")

    train = subparsers.add_parser("train", help="Fine-tune LayoutLMv3 for field token classification")
    train.add_argument("--prepared-manifest", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--base-model", default="microsoft/layoutlmv3-base")
    train.add_argument("--epochs", type=float, default=3.0)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--learning-rate", type=float, default=5e-5)

    predict = subparsers.add_parser("predict", help="Return final bounding boxes for requested fields")
    predict.add_argument("--image", required=True)
    predict.add_argument("--fields", required=True)
    predict.add_argument("--model-dir")
    predict.add_argument("--lang", default="en")

    ground = subparsers.add_parser("ground", help="Run graph-based, row-aware key-value grounding")
    ground.add_argument("--image", required=True)
    ground.add_argument("--fields", required=True)
    ground.add_argument("--tables", help="Optional table structure JSON/CSV")
    ground.add_argument("--ocr-json", help="Optional OCR words JSON for Python 3.14 or offline OCR")
    ground.add_argument("--ocr-backend", choices=["paddle", "none"], default="paddle")
    ground.add_argument("--output-graph", help="Optional path to write the document graph JSON")
    ground.add_argument("--lang", default="en")


    preprocess = subparsers.add_parser(
        "preprocess-annotation",
        help="Convert dataset annotation JSON to DocuXray requested-fields JSON",
    )
    preprocess.add_argument("--input", required=True)
    preprocess.add_argument("--output", required=True)
    preprocess.add_argument("--key-mode", choices=["label", "path"], default="label")
    preprocess.add_argument("--source", choices=["auto", "field_metadata", "postprocessed", "generic"], default="auto")
    preprocess.add_argument("--include-null", action="store_true")
    preprocess.add_argument("--include-booleans", action="store_true")

    args = parser.parse_args()
    if args.command == "prepare":
        prepare_manifest(args.manifest, args.output, lang=args.lang)
    elif args.command == "train":
        train_model(
            args.prepared_manifest,
            args.output_dir,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    elif args.command == "predict":
        results = predict_fields(
            args.image,
            read_fields(args.fields),
            model_dir=args.model_dir,
            lang=args.lang,
        )
        print(json.dumps([result.to_dict() for result in results], indent=2))
    elif args.command == "ground":
        results = run_grounding_pipeline(
            args.image,
            args.fields,
            tables_path=args.tables,
            ocr_json_path=args.ocr_json,
            ocr_backend=args.ocr_backend,
            output_graph_path=args.output_graph,
            lang=args.lang,
        )
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        convert_path(
            args.input,
            args.output,
            key_mode=args.key_mode,
            source=args.source,
            include_null=args.include_null,
            include_booleans=args.include_booleans,
        )


if __name__ == "__main__":
    main()
