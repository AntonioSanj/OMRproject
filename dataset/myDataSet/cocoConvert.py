import pandas as pd
import json
import os

from constants import *

# Directory containing CSV files
csv_directory = myDataCsv2
output_file = myDataCoco2

# Category mapping
"""
category_mapping = {
    "One": 1,
    "Double": 2,
    "Four": 3,
    "Half": 4,
    "Quarter": 5,
    "GClef": 6,
    "FClef": 7,
    "OpeningBracket": 8,
    "RestOne": 9,
    "RestHalf": 10,
}

"""
category_mapping = {
    "One": 1,
    "Double": 2,
    "Four": 3,
    "Half": 4,
    "Quarter": 5
}

# Supercategory name
supercategory_name = "figure"

# Initialize COCO format structure
coco_format = {
    "annotations": [],
    "categories": [
        {"id": 0, "name": supercategory_name, "supercategory": "none"}
    ] + [
        {"id": cid, "name": name, "supercategory": supercategory_name}
        for name, cid in category_mapping.items()
    ],
    "images": []
}

annotation_id = 0  # Global counter for unique annotation IDs
image_id = 0  # Counter for unique image IDs

# Process each CSV file
for csv_file in os.listdir(csv_directory):
    if csv_file.endswith(".csv"):
        # Parse the CSV
        file_path = os.path.join(csv_directory, csv_file)
        data = pd.read_csv(file_path)

        # Extract image name (assuming it matches the CSV file)
        image_name = os.path.splitext(csv_file)[0] + ".png"

        # Add image entry
        coco_format["images"].append({"id": image_id, "file_name": image_name})

        # Add annotations
        for _, row in data.iterrows():
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            class_title = row["classTitle"]

            # transformations
            x1 = x1 - round((x2-x1) * 0.2)
            x2 = x2 + round((x2-x1) * 0.2)
            y1 = y1 - round((y2-y1) * 0.2)
            y2 = y2 + round((y2-y1) * 0.2)

            # Map classTitle to category_id
            category_id = category_mapping.get(class_title)
            if category_id is None:
                raise ValueError(f"Unknown category: {class_title}")

            width = x2 - x1
            height = y2 - y1
            segmentation = [x1, y1, x2, y1, x2, y2, x1, y2]  # Approximate polygon
            area = width * height

            annotation = {
                "segmentation": [segmentation],
                "area": area,
                "bbox": [x1, y1, width, height],
                "iscrowd": 0,
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1

        image_id += 1

# Save to JSON file
with open(output_file, "w") as f:
    json.dump(coco_format, f, indent=4)

print(f"COCO JSON saved to {output_file}")
