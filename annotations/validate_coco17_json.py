import json

from pathlib import Path

# 🔹 Define your JSON path here

INPUT_JSON = Path(r"C:\Users\220250572\OneDrive - Regal Rexnord\Desktop\Ergonmics\ergoposekeypoints\annotations\person_keypoints_coco17_fixed.json")

# Standard COCO-17 order

COCO_KEYPOINTS = [

    "nose", "left_eye", "right_eye", "left_ear", "right_ear",

    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",

    "left_wrist", "right_wrist", "left_hip", "right_hip",

    "left_knee", "right_knee", "left_ankle", "right_ankle"

]

def main():

    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))

    categories = data.get("categories", [])

    images = data.get("images", [])

    anns = data.get("annotations", [])

    errors = []

    # --- Category checks ---

    if len(categories) != 1:

        errors.append(f"Expected 1 category, found {len(categories)}")

    if categories:

        kp_names = categories[0].get("keypoints", [])

        if kp_names != COCO_KEYPOINTS:

            errors.append("Keypoint names/order do not match COCO-17 standard")

    # --- Annotation checks ---

    for idx, ann in enumerate(anns):

        if "bbox" not in ann or len(ann["bbox"]) != 4:

            errors.append(f"Annotation {ann.get('id', idx)} missing bbox")

        kps = ann.get("keypoints", [])

        if len(kps) != 51:

            errors.append(f"Annotation {ann.get('id', idx)} has {len(kps)} keypoints (expected 51)")

        else:

            # check v values

            vs = kps[2::3]  # every 3rd value

            if not all(v in [0,1,2] for v in vs):

                errors.append(f"Annotation {ann.get('id', idx)} has invalid visibility flag(s): {vs}")

            # check num_keypoints

            expected_num = sum(1 for v in vs if v > 0)

            if ann.get("num_keypoints") != expected_num:

                errors.append(

                    f"Annotation {ann.get('id', idx)} num_keypoints={ann.get('num_keypoints')} "

                    f"but counted {expected_num}"

                )

    # --- Report ---

    print("=== COCO-17 Validation Report ===")

    print(f"Images: {len(images)} | Annotations: {len(anns)} | Categories: {len(categories)}")

    if errors:

        print("\n❌ Issues found:")

        for e in errors:

            print(" -", e)

    else:

        print("\n✅ All checks passed. JSON is COCO-17 compliant.")

if __name__ == "__main__":

    main()
 