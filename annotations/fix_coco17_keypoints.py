import json

from pathlib import Path

# -------------------

# 🔹 Define your file paths here

INPUT_JSON = Path(r"C:\Users\220250572\OneDrive - Regal Rexnord\Desktop\Ergonmics\ergoposekeypoints\annotations\person_keypoints_default.json")       # input file

OUTPUT_JSON = Path(r"C:\Users\220250572\OneDrive - Regal Rexnord\Desktop\Ergonmics\ergoposekeypoints\annotations\person_keypoints_coco17_fixed.json") # corrected output

# -------------------

# Standard COCO-17 order

COCO_KEYPOINTS = [

    "nose", "left_eye", "right_eye", "left_ear", "right_ear",

    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",

    "left_wrist", "right_wrist", "left_hip", "right_hip",

    "left_knee", "right_knee", "left_ankle", "right_ankle"

]

# Load JSON

data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

categories = data.get("categories", [])

if not categories:

    raise ValueError("No categories found in JSON")

# Assume only one category ("person")

cat = categories[0]

old_kps = cat.get("keypoints", [])

print("Old keypoints order:", old_kps)

# Build mapping old_index -> new_index

mapping = {}

for i, name in enumerate(old_kps):

    if name in COCO_KEYPOINTS:

        mapping[i] = COCO_KEYPOINTS.index(name)

    else:

        print(f"⚠️ Warning: keypoint '{name}' not in COCO-17 list, skipping.")

print("Remap dictionary (old->new):", mapping)

# Update category keypoints & skeleton

cat["keypoints"] = COCO_KEYPOINTS

# Reset skeleton to standard COCO pairs

cat["skeleton"] = [

    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],

    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],

    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],

    [2, 4], [3, 5]

]

# Fix annotations

for ann in data.get("annotations", []):

    kps = ann.get("keypoints", [])

    if len(kps) % 3 != 0:

        continue  # skip invalid

    num_old = len(kps) // 3

    new_kps = [0] * (len(COCO_KEYPOINTS) * 3)

    for old_idx in range(num_old):

        if old_idx not in mapping:

            continue

        new_idx = mapping[old_idx]

        x, y, v = kps[old_idx*3: old_idx*3+3]

        new_kps[new_idx*3: new_idx*3+3] = [x, y, v]

    ann["keypoints"] = new_kps

    ann["num_keypoints"] = sum(1 for i in range(2, len(new_kps), 3) if new_kps[i] > 0)

# Save corrected file

OUTPUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

print(f"✅ Fixed JSON saved to: {OUTPUT_JSON}")
 