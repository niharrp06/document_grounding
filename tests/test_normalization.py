from docuxray.normalization import normalize_field_requests, normalize_value


def test_normalize_field_requests_preserves_duplicate_keys() -> None:
    requests = normalize_field_requests(
        {
            "fields": [
                {"field_id": "a0", "key": "Amount", "value": "5000", "row_index": 0},
                {"field_id": "a1", "key": "Amount", "value": "7000", "row_index": 1},
            ]
        }
    )
    assert [request.field_id for request in requests] == ["a0", "a1"]
    assert [request.row_index for request in requests] == [0, 1]


def test_normalize_value_handles_currency() -> None:
    assert normalize_value("₹ 5,000.00", "currency") == "5000.00"
