from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from .types import FieldRequest

_NON_WORD = re.compile(r"\W+", flags=re.UNICODE)
_NUMBER_CHARS = re.compile(r"[^0-9.\-]")


def normalize_text(text: str) -> str:
    return _NON_WORD.sub("", text).casefold()


def normalize_value(value: str, value_type: str | None = None) -> str:
    if value_type in {"number", "currency"} or _looks_numeric(value):
        numeric = _NUMBER_CHARS.sub("", value)
        if numeric:
            return numeric
    return normalize_text(value)


def normalize_field_requests(data: Any) -> list[FieldRequest]:
    if isinstance(data, dict) and "fields" in data:
        return [_request_from_item(item, index) for index, item in enumerate(data["fields"])]
    if isinstance(data, dict) and "records" in data:
        return list(_requests_from_records(data["records"]))
    if isinstance(data, dict) and data:
        return [
            FieldRequest(field_id=str(key), key=str(key), value=str(value))
            for key, value in data.items()
        ]
    raise ValueError("Field request JSON must be a non-empty object with fields, records, or key values")


def _request_from_item(item: Any, index: int) -> FieldRequest:
    if not isinstance(item, dict):
        raise ValueError("Every fields[] item must be an object")
    key = item.get("key") or item.get("field")
    value = item.get("value")
    if not isinstance(key, str) or not isinstance(value, str):
        raise ValueError("Every field request needs string key and value properties")
    field_id = str(item.get("field_id") or f"{normalize_text(key) or 'field'}_{index}")
    row_index = item.get("row_index")
    if row_index is not None and not isinstance(row_index, int):
        raise ValueError("row_index must be an integer when provided")
    return FieldRequest(
        field_id=field_id,
        key=key,
        value=value,
        row_index=row_index,
        record_id=item.get("record_id"),
        value_type=item.get("value_type"),
        metadata={k: v for k, v in item.items() if k not in _FIELD_KEYS},
    )


def _requests_from_records(records: Any) -> Iterable[FieldRequest]:
    if not isinstance(records, list):
        raise ValueError("records must be a list")
    for record_index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError("Every record must be an object")
        fields = record.get("fields", record)
        if not isinstance(fields, dict):
            raise ValueError("Record fields must be an object")
        row_index = record.get("row_index")
        record_id = str(record.get("record_id") or f"record_{record_index}")
        for field_index, (key, value) in enumerate(fields.items()):
            if key in {"row_index", "record_id"}:
                continue
            if not isinstance(value, str):
                raise ValueError("Record field values must be strings")
            yield FieldRequest(
                field_id=f"{record_id}_{normalize_text(str(key)) or field_index}",
                key=str(key),
                value=value,
                row_index=row_index if isinstance(row_index, int) else None,
                record_id=record_id,
            )


def _looks_numeric(value: str) -> bool:
    digits = sum(ch.isdigit() for ch in value)
    return digits > 0 and digits >= max(1, len(value.strip()) // 2)


_FIELD_KEYS = {"field_id", "field", "key", "value", "row_index", "record_id", "value_type"}
