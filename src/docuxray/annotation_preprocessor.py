from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

FIELD_ID_CHARS = re.compile(r"[^A-Za-z0-9_]+")
LINE_ITEM_PATH = re.compile(r"^lineItems\.(\d+)\.(.+)$")

LABEL_MAP = {
    "currency": "Currency",
    "invoiceStatus": "Invoice Status",
    "invoiceInfo.documentNumber": "Invoice No",
    "invoiceInfo.issueDate": "Date",
    "invoiceInfo.issueDateISO": "Date",
    "invoiceInfo.dueDate": "Due Date",
    "invoiceInfo.dueDateISO": "Due Date",
    "invoiceInfo.purchaseOrderNumber": "Purchase Order Number",
    "invoiceInfo.customerNumber": "Customer Number",
    "invoiceInfo.paymentTerms": "Payment terms",
    "invoiceInfo.customerMemo": "Re",
    "parties.seller.name": "Seller Name",
    "parties.seller.phone": "Seller Phone",
    "parties.seller.email": "Seller Email",
    "parties.seller.addressStructured.address": "Seller Address",
    "parties.seller.addressStructured.city": "Seller City",
    "parties.seller.addressStructured.postal_code": "Seller Postal Code",
    "parties.customer.name": "Customer Name",
    "parties.customer.phone": "Customer Phone",
    "parties.customer.email": "Customer Email",
    "parties.customer.addressStructured.address": "Customer Address",
    "parties.customer.addressStructured.city": "Customer City",
    "parties.customer.addressStructured.postal_code": "Customer Postal Code",
    "parties.shipTo.name": "Ship To",
    "parties.shipTo.addressStructured.address": "Ship To Address",
    "parties.shipTo.addressStructured.city": "Ship To City",
    "parties.shipTo.addressStructured.postal_code": "Ship To Postal Code",
    "shippingInfo.carrier": "Carrier",
    "shippingInfo.trackingNumber": "Tracking Number",
    "shippingInfo.deliveryDate": "Delivery Date",
    "shippingInfo.deliveryDateISO": "Delivery Date",
    "totals.discountTotal.originalValue": "Discount",
    "totals.subtotal.originalValue": "Subtotal",
    "totals.totalExcludingTax.originalValue": "Total Excluding Tax",
    "totals.taxAmount.originalValue": "Tax",
    "totals.taxTotal.originalValue": "Tax",
    "totals.totalIncludingTax.originalValue": "Total",
    "totals.amountPaid.originalValue": "Amount Paid",
    "totals.balanceDue.originalValue": "Balance Due",
    "lineItems.*.description": "Description",
    "lineItems.*.itemCode": "Item Code",
    "lineItems.*.quantity.originalValue": "Quantity",
    "lineItems.*.unitPrice.originalValue": "Unit Price",
    "lineItems.*.lineTotalIncludingTax.originalValue": "Total",
    "lineItems.*.lineTotalExcludingTax.originalValue": "Line Total",
    "lineItems.*.lineTaxAmount.originalValue": "Tax",
    "lineItems.*.lineTaxPercent.originalValue": "Tax Percent",
}

VALUE_KEYS = ("final_value", "value", "text", "content", "answer", "originalValue")
KEY_KEYS = ("key", "field", "name", "label", "path", "source_path")


def convert_annotation(
    annotation: Any,
    *,
    key_mode: str = "label",
    source: str = "auto",
    include_null: bool = False,
    include_booleans: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Convert common invoice annotation JSON variants into DocuXray fields JSON.

    The converter is deliberately defensive: unknown nested dictionaries are flattened,
    nulls/booleans are skipped by default, and lineItems.N.* paths receive row_index=N.
    """
    if key_mode not in {"label", "path"}:
        raise ValueError("key_mode must be either 'label' or 'path'")

    fields = []
    seen_ids: dict[str, int] = {}
    for entry in iter_annotation_entries(annotation, source=source):
        path = str(entry.get("path") or "field").strip()
        if not path:
            continue
        value = repair_text(entry.get("value"))
        if not should_include(value, include_null=include_null, include_booleans=include_booleans):
            continue

        base_field_id = field_id_from_path(path) or "field"
        field_id = unique_field_id(base_field_id, seen_ids)
        field = {
            "field_id": field_id,
            "key": path if key_mode == "path" else label_for_path(path),
            "value": stringify_value(value),
            "metadata": {
                "source_path": path,
                "selected_from": entry.get("selected_from"),
            },
        }
        row_index = row_index_from_path(path)
        if row_index is not None:
            field["row_index"] = row_index
        fields.append(field)
    return {"fields": fields}


def iter_annotation_entries(annotation: Any, *, source: str) -> Iterable[dict[str, Any]]:
    if source not in {"auto", "field_metadata", "postprocessed", "generic"}:
        raise ValueError("source must be one of: auto, field_metadata, postprocessed, generic")

    if isinstance(annotation, dict) and isinstance(annotation.get("fields"), list):
        yield from iter_existing_fields(annotation["fields"])
        return

    if source in {"auto", "field_metadata"}:
        metadata = get_nested(annotation, ["annotation_meta", "field_metadata"])
        if isinstance(metadata, list) and metadata:
            yield from iter_key_value_items(metadata, default_selected_from_key="selected_from")
            return

    if source in {"auto", "postprocessed"}:
        postprocessed = annotation.get("postprocessed") if isinstance(annotation, dict) else None
        if isinstance(postprocessed, (dict, list)):
            yield from iter_flattened(postprocessed)
            return

    if isinstance(annotation, list):
        yield from iter_key_value_items(annotation)
        return

    if isinstance(annotation, dict):
        for known_key in ("annotations", "labels", "items", "data", "results"):
            value = annotation.get(known_key)
            if isinstance(value, list):
                yield from iter_key_value_items(value)
                return
        yield from iter_flattened(annotation)


def iter_existing_fields(fields: list[Any]) -> Iterable[dict[str, Any]]:
    for index, item in enumerate(fields):
        if not isinstance(item, dict):
            continue
        key = item.get("key") or item.get("field") or item.get("name") or f"field_{index}"
        value = item.get("value")
        yield {
            "path": str(item.get("metadata", {}).get("source_path") or key),
            "value": value,
            "selected_from": item.get("metadata", {}).get("selected_from"),
        }


def iter_key_value_items(items: list[Any], *, default_selected_from_key: str | None = None) -> Iterable[dict[str, Any]]:
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            yield {"path": f"field_{index}", "value": item, "selected_from": None}
            continue
        path = first_present(item, KEY_KEYS) or f"field_{index}"
        value = first_present(item, VALUE_KEYS)
        selected_from = item.get(default_selected_from_key) if default_selected_from_key else item.get("selected_from")
        if value is None and any(isinstance(item.get(key), (dict, list)) for key in VALUE_KEYS):
            value = first_present(item, VALUE_KEYS)
        yield {"path": str(path), "value": value, "selected_from": selected_from}


def iter_flattened(value: Any) -> Iterable[dict[str, Any]]:
    for path, child in flatten(value):
        yield {"path": path, "value": child, "selected_from": "postprocessed"}


def flatten(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        if set(value.keys()) == {"originalValue"}:
            yield prefix, value.get("originalValue")
            return
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from flatten(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            yield from flatten(child, child_prefix)
    else:
        yield prefix, value


def label_for_path(path: str) -> str:
    if path in LABEL_MAP:
        return LABEL_MAP[path]
    wildcard_path = LINE_ITEM_PATH.sub(r"lineItems.*.\2", path)
    if wildcard_path in LABEL_MAP:
        return LABEL_MAP[wildcard_path]
    leaf = path.split(".")[-1]
    if leaf == "originalValue" and len(path.split(".")) >= 2:
        leaf = path.split(".")[-2]
    return humanize(leaf)


def humanize(value: str) -> str:
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)
    value = value.replace("_", " ").replace("-", " ")
    return value.strip().title()


def field_id_from_path(path: str) -> str:
    return FIELD_ID_CHARS.sub("_", path).strip("_").lower()


def unique_field_id(base: str, seen_ids: dict[str, int]) -> str:
    count = seen_ids.get(base, 0)
    seen_ids[base] = count + 1
    return base if count == 0 else f"{base}_{count}"


def row_index_from_path(path: str) -> int | None:
    match = LINE_ITEM_PATH.match(path)
    return int(match.group(1)) if match else None


def should_include(value: Any, *, include_null: bool, include_booleans: bool) -> bool:
    if value is None:
        return include_null
    if isinstance(value, bool):
        return include_booleans
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (dict, list)):
        return False
    return True


def stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def repair_text(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if chr(321) in value:
        value = value.replace(chr(321), chr(163))
    if "\u00c2" in value:
        try:
            return value.encode("latin1").decode("utf-8")
        except UnicodeError:
            return value.replace(chr(194) + chr(163), chr(163))
    return value


def first_present(item: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in item:
            return item[key]
    return None


def get_nested(value: Any, keys: list[str]) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    key_mode: str = "label",
    source: str = "auto",
    include_null: bool = False,
    include_booleans: bool = False,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    converted = convert_annotation(
        load_json(input_path),
        key_mode=key_mode,
        source=source,
        include_null=include_null,
        include_booleans=include_booleans,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(converted, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert_path(
    input_path: str | Path,
    output_path: str | Path,
    *,
    key_mode: str = "label",
    source: str = "auto",
    include_null: bool = False,
    include_booleans: bool = False,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        for source_file in sorted(input_path.glob("*.json")):
            convert_file(
                source_file,
                output_path / source_file.name,
                key_mode=key_mode,
                source=source,
                include_null=include_null,
                include_booleans=include_booleans,
            )
    else:
        convert_file(
            input_path,
            output_path,
            key_mode=key_mode,
            source=source,
            include_null=include_null,
            include_booleans=include_booleans,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert annotation JSON files into DocuXray requested-fields JSON."
    )
    parser.add_argument("--input", required=True, help="Source annotation JSON file or directory")
    parser.add_argument("--output", required=True, help="Output fields JSON file or directory")
    parser.add_argument("--key-mode", choices=["label", "path"], default="label")
    parser.add_argument(
        "--source",
        choices=["auto", "field_metadata", "postprocessed", "generic"],
        default="auto",
    )
    parser.add_argument("--include-null", action="store_true")
    parser.add_argument("--include-booleans", action="store_true")
    args = parser.parse_args()
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
