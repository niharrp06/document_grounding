from docuxray.dataset import build_label_maps


def test_build_label_maps_puts_outside_label_first() -> None:
    records = [{"labels": ["O", "I-name", "B-name"]}, {"labels": ["B-id"]}]
    label2id, id2label = build_label_maps(records)
    assert label2id == {"O": 0, "B-id": 1, "B-name": 2, "I-name": 3}
    assert id2label[0] == "O"
