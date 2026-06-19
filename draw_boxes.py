import cv2
import json
import os


# Paths

IMAGE_PATH = r"C:\Users\Nihar Pathak\OneDrive\Desktop\DocuXrayNew\examples\sample_invoice.png"
DOCUMENT_JSON = r"C:\Users\Nihar Pathak\OneDrive\Desktop\DocuXrayNew\outputs\sample_document_graph.json"
FIELDS_JSON = r"C:\Users\Nihar Pathak\OneDrive\Desktop\DocuXrayNew\examples\sample_fields.json"


# Load image

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )


# Load JSON files

with open(DOCUMENT_JSON, "r", encoding="utf-8") as f:
    doc = json.load(f)

with open(FIELDS_JSON, "r", encoding="utf-8") as f:
    targets = json.load(f)


# Draw matched boxes

for field in targets["fields"]:

    target_key = field["key"]
    target_value = str(field["value"])
    target_row = field["row_index"]

    found = False

    for cell in doc["cells"]:

        if (
            cell["header_text"] == target_key
            and str(cell["text"]) == target_value
            and cell["row_index"] == target_row
        ):

            x1, y1, x2, y2 = cell["box"]

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            label = (
                f"{target_key}"
                f"={target_value}"
                f" (row {target_row})"
            )

            # cv2.putText(
            #     image,
            #     label,
            #     (x1, max(y1 - 10, 0)),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     0.5,
            #     (0, 0, 255),
            #     1
            # )

            found = True
            break

    if not found:
        print(
            f"WARNING: No match found for "
            f"{target_key}={target_value}, "
            f"row={target_row}"
        )


# Save output

os.makedirs("outputs", exist_ok=True)

output_path = os.path.join(
    "outputs",
    "highlighted_fields.jpg"
)

cv2.imwrite(output_path, image)

print(f"Saved output to: {output_path}")