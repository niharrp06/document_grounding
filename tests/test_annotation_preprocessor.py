from docuxray.annotation_preprocessor import convert_annotation


def test_annotation_preprocessor_uses_field_metadata_and_repairs_currency() -> None:
    annotation = {
        "postprocessed": {},
        "annotation_meta": {
            "field_metadata": [
                {"key": "invoiceInfo.documentNumber", "final_value": 2921, "selected_from": "GPT"},
                {
                    "key": "totals.totalIncludingTax.originalValue",
                    "final_value": chr(194) + chr(163) + "70.00",
                    "selected_from": "manual_entry",
                },
                {"key": "invoiceInfo.dueDate", "final_value": None, "selected_from": "GPT"},
            ]
        },
    }

    converted = convert_annotation(annotation)

    assert converted["fields"] == [
        {
            "field_id": "invoiceinfo_documentnumber",
            "key": "Invoice No",
            "value": "2921",
            "metadata": {"source_path": "invoiceInfo.documentNumber", "selected_from": "GPT"},
        },
        {
            "field_id": "totals_totalincludingtax_originalvalue",
            "key": "Total",
            "value": "\u00a370.00",
            "metadata": {
                "source_path": "totals.totalIncludingTax.originalValue",
                "selected_from": "manual_entry",
            },
        },
    ]


def test_annotation_preprocessor_adds_line_item_row_index() -> None:
    annotation = {
        "annotation_meta": {
            "field_metadata": [
                {
                    "key": "lineItems.1.description",
                    "final_value": "Carry Out Landlords Gas Safety Inspection",
                }
            ]
        }
    }

    converted = convert_annotation(annotation)

    assert converted["fields"][0]["key"] == "Description"
    assert converted["fields"][0]["row_index"] == 1


def test_annotation_preprocessor_flattens_postprocessed_when_metadata_missing() -> None:
    annotation = {
        "postprocessed": {
            "invoiceInfo": {"documentNumber": 2921},
            "lineItems": [{"description": "Service Boiler"}],
        }
    }

    converted = convert_annotation(annotation)

    assert converted["fields"][0]["key"] == "Invoice No"
    assert converted["fields"][1]["row_index"] == 0


def test_annotation_preprocessor_accepts_generic_list_items() -> None:
    annotation = [{"label": "Invoice No", "value": "2921"}]

    converted = convert_annotation(annotation, source="generic")

    assert converted["fields"] == [
        {
            "field_id": "invoice_no",
            "key": "Invoice No",
            "value": "2921",
            "metadata": {"source_path": "Invoice No", "selected_from": None},
        }
    ]
