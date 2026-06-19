import cv2
import json
import os

# Paths
image_path = r"examples\sample_invoice.png"
json_path = r"outputs\sample_document_graph.json"

# Load image
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Could not load image: {image_path}")

# Load JSON
with open(json_path, "r") as f:
    data = json.load(f)

# Draw all bounding boxes
for field_name, field_info in data.items():

    bbox = field_info["bbox"]  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = bbox

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        field_name,
        (x1, max(y1 - 10, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

# Ensure output directory exists
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# Save image
output_path = os.path.join(output_dir, "document_with_boxes.jpg")

cv2.imwrite(output_path, image)

print(f"Image saved to: {output_path}")