import os
import cv2
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from ultralytics import YOLO

# ----------- CONFIGURATION -----------
image_path = r"C:/Users/220250572/OneDrive - Regal Rexnord/Desktop/Ergonmics/test_images/test_image_before.jpg"  # Replace with your image path
output_folder = r"C:/Users/220250572/OneDrive - Regal Rexnord/Desktop/Ergonmics/output/images"
os.makedirs(output_folder, exist_ok=True)
model_path = r"C:/Users/220250572/OneDrive - Regal Rexnord/Desktop/Ergonmics/runs/pose/train/weights/best.pt"  # Replace with your YOLO pose model path
excel_path = os.path.join(output_folder, "Ergonomic_Report.xlsx")
pdf_path = os.path.join(output_folder, "Ergonomic_Report.pdf")
annotated_img_path = os.path.join(output_folder, "pose_skeleton.jpg")
visualization_path = os.path.join(output_folder, "ergonomic_report_visual.png")

# ----------- FUNCTIONS -----------
EPS = 1e-8
# -----------------------
# Core angle utilities
# -----------------------
def angle_at_joint(p1, p2, p3):
   """Return internal angle at p2 formed by p1-p2-p3 in degrees (0..180)."""
   a = np.array(p1, dtype=float) - np.array(p2, dtype=float)
   b = np.array(p3, dtype=float) - np.array(p2, dtype=float)
   denom = (np.linalg.norm(a) * np.linalg.norm(b) + EPS)
   cosang = np.dot(a, b) / denom
   cosang = np.clip(cosang, -1.0, 1.0)
   return float(np.degrees(np.arccos(cosang)))
def vector(p_from, p_to):
   return np.array(p_to, dtype=float) - np.array(p_from, dtype=float)
def angle_between_vectors(v1, v2):
   denom = (np.linalg.norm(v1) * np.linalg.norm(v2) + EPS)
   cosang = np.dot(v1, v2) / denom
   cosang = np.clip(cosang, -1.0, 1.0)
   return float(np.degrees(np.arccos(cosang)))
def angle_to_vertical(p_top, p_bottom, image_y_increases_down=True):
   """
   Angle between p_top->p_bottom and vertical axis (image coords).
   Returns 0° when vector is perfectly vertical (upright trunk).
   """
   v = vector(p_top, p_bottom)
   vertical = np.array([0., 1.0]) if image_y_increases_down else np.array([0., -1.0])
   return angle_between_vectors(v, vertical)
# -----------------------
# Keypoint -> angle extractor
# -----------------------
def extract_angles_from_keypoints_flat(keypoints_flat):
   """
   Input: flattened list/array of 17*(x,y,conf) (COCO-17 order).
   Output: dictionary of angles (trunk, neck, upper_arm, lower_arm, wrist, leg)
   NOTE: uses left-side joints (left_shoulder/left_elbow/left_wrist, left_hip/left_knee/left_ankle).
   """
   # assumed ordering (x,y,conf) repeated for:
   # nose, left_eye, right_eye, left_ear, right_ear,
   # left_shoulder, right_shoulder, left_elbow, right_elbow,
   # left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle
   names = [
       'nose','left_eye','right_eye','left_ear','right_ear',
       'left_shoulder','right_shoulder','left_elbow','right_elbow',
       'left_wrist','right_wrist','left_hip','right_hip','left_knee','right_knee','left_ankle','right_ankle'
   ]
   arr = np.array(keypoints_flat, dtype=float)
   kp = {}
   for i,name in enumerate(names):
       x = float(arr[i*3])
       y = float(arr[i*3 + 1])
       kp[name] = (x,y)
   # compute midpoints (use whole torso direction rather than single side)
   shoulder_mid = ((kp['left_shoulder'][0] + kp['right_shoulder'][0]) / 2.0,
                   (kp['left_shoulder'][1] + kp['right_shoulder'][1]) / 2.0)
   hip_mid = ((kp['left_hip'][0] + kp['right_hip'][0]) / 2.0,
              (kp['left_hip'][1] + kp['right_hip'][1]) / 2.0)
   # trunk: shoulder_mid -> hip_mid relative to vertical
   trunk_ang = angle_to_vertical(shoulder_mid, hip_mid)
   # neck: angle between head->shoulder_mid and trunk_vector (shoulder_mid->hip_mid)
   head_point = kp.get('nose', shoulder_mid)
   neck_ang = angle_between_vectors(vector(head_point, shoulder_mid), vector(shoulder_mid, hip_mid))
   # upper arm: angle between shoulder_mid->left_elbow and trunk vector
   upper_arm_ang = angle_between_vectors(vector(shoulder_mid, kp['left_elbow']), vector(shoulder_mid, hip_mid))
   # lower arm: elbow internal angle (shoulder->elbow->wrist)
   lower_arm_ang = angle_at_joint(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])
   # wrist: if finger keypoints unavailable, use forearm (elbow->wrist) vs vertical as proxy
   wrist_ang = angle_between_vectors(vector(kp['left_elbow'], kp['left_wrist']), np.array([0., 1.0]))
   # leg: knee internal angle (hip->knee->ankle)
   leg_ang = angle_at_joint(kp['left_hip'], kp['left_knee'], kp['left_ankle'])
   return {
       "trunk": float(trunk_ang),
       "neck": float(neck_ang),
       "upper_arm": float(upper_arm_ang),
       "lower_arm": float(lower_arm_ang),
       "wrist": float(wrist_ang),
       "leg": float(leg_ang),
       "kp": kp
   }
# -----------------------
# Simple bucket mapping (adjust thresholds if you want official tables)
# -----------------------
def map_reba_buckets(angles):
   trunk_score = 1 if angles['trunk'] <= 10 else 2 if angles['trunk'] <= 20 else 3
   neck_score = 1 if angles['neck'] <= 10 else 2 if angles['neck'] <= 20 else 3
   leg_score = 1 if angles['leg'] <= 30 else 2 if angles['leg'] <= 60 else 3
   upper_arm_score = 1 if angles['upper_arm'] <= 20 else 2 if angles['upper_arm'] <= 60 else 3
   # lower arm assumed internal angle: straight ~180 -> low score
   la = angles['lower_arm']
   lower_arm_score = 1 if la >= 150 or la <= 30 else 2 if la >= 60 else 3
   wrist_score = 1 if angles['wrist'] <= 15 else 2 if angles['wrist'] <= 30 else 3
   return {
       "Trunk": trunk_score,
       "Neck": neck_score,
       "Leg": leg_score,
       "Upper Arm": upper_arm_score,
       "Lower Arm": lower_arm_score,
       "Wrist": wrist_score
   }
def map_rula_buckets(angles):
   upper_arm_score = 1 if angles['upper_arm'] <= 20 else 2 if angles['upper_arm'] <= 60 else 3
   la = angles['lower_arm']
   lower_arm_score = 1 if la >= 150 or la <= 30 else 2 if la <= 120 else 3
   wrist_score = 1 if angles['wrist'] <= 15 else 2 if angles['wrist'] <= 30 else 3
   neck_score = 1 if angles['neck'] <= 10 else 2 if angles['neck'] <= 20 else 3
   trunk_score = 1 if angles['trunk'] <= 10 else 2 if angles['trunk'] <= 20 else 3
   return {
       "Upper Arm": upper_arm_score,
       "Lower Arm": lower_arm_score,
       "Wrist": wrist_score,
       "Neck": neck_score,
       "Trunk": trunk_score
   }
# -----------------------
# REBA / RULA evaluate functions (drop-in)
# -----------------------
def evaluate_reba_corrected(keypoints_flat, load_force_score=0, activity_score=0, coupling_score=1):
   """
   Simplified REBA-style evaluation using corrected angles.
   coupling_score default kept 1 for 'good'.
   load_force_score: 0/1/2 as you used; activity_score: 0/1/2 as before.
   """
   angles = extract_angles_from_keypoints_flat(keypoints_flat)
   joint_scores = map_reba_buckets(angles)
   total = sum(joint_scores.values()) + int(coupling_score) + int(load_force_score) + int(activity_score)
   if total <= 3:
       inference = "Low risk"
   elif total <= 7:
       inference = "Moderate risk"
   elif total <= 10:
       inference = "High risk"
   else:
       inference = "Very high risk"
   return {
       "angles": {k: round(v,2) for k,v in angles.items() if k in ['trunk','neck','upper_arm','lower_arm','wrist','leg']},
       "scores": joint_scores,
       "Coupling": coupling_score,
       "Load/Force": load_force_score,
       "Activity": activity_score,
       "total_score": int(total),
       "inference": inference
   }
def evaluate_rula_corrected(keypoints_flat):
   angles = extract_angles_from_keypoints_flat(keypoints_flat)
   joint_scores = map_rula_buckets(angles)
   total = sum(joint_scores.values())
   if total <= 3:
       inference = "Acceptable posture"
   elif total <= 5:
       inference = "Further investigation"
   elif total <= 7:
       inference = "Changes soon"
   else:
       inference = "Immediate changes"
   return {
       "angles": {k: round(v,2) for k,v in angles.items() if k in ['trunk','neck','upper_arm','lower_arm','wrist','leg']},
       "scores": joint_scores,
       "total_score": int(total),
       "inference": inference
   }

def evaluate_niosh(load_weight, H, V, D, F, A, C):
    LC = 23
    HM = 25 / H if H > 0 else 0
    VM = 1 - 0.003 * abs(V - 75)
    DM = 0.82 + 4.5 / D if D > 0 else 0
    AM = 1 - 0.0032 * A
    FM = 0.94 if F < 0.2 else 0.88 if F < 0.5 else 0.75
    CM = 1 if C == "good" else 0.95 if C == "fair" else 0.9
    RWL = LC * HM * VM * DM * AM * FM * CM
    LI = load_weight / RWL if RWL > 0 else 0
    inference = "Safe lifting task." if LI <= 1 else "Unsafe lifting task. Ergonomic improvements needed."
    return {
        "RWL": round(RWL, 2),
        "LI": round(LI, 2),
        "inference": inference
    }

def draw_dynamic_visualization(keypoints, reba_scores, reba_total, rula_total, output_path):
    img = np.ones((400, 300, 3), dtype=np.uint8) * 255
    color_map = {1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}

    regions = {
        'Neck': (130, 50, 40, 30),
        'Trunk': (120, 80, 60, 80),
        'Upper Arm': (80, 80, 40, 60),
        'Lower Arm': (60, 140, 40, 60),
        'Wrist': (50, 200, 30, 30),
        'Leg': (120, 160, 60, 100)
    }
    for part, (x, y, w, h) in regions.items():
        score = reba_scores.get(part, 1)
        cv2.rectangle(img, (x, y), (x+w, y+h), color_map[score], -1)
        cv2.putText(img, part, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Skeleton overlay (scaled and centered)
    pairs = [(5, 7), (7, 9), (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16)]
    for p1, p2 in pairs:
        x1, y1 = int(keypoints[p1*3] / 4), int(keypoints[p1*3+1] / 4)
        x2, y2 = int(keypoints[p2*3] / 4), int(keypoints[p2*3+1] / 4)
        cv2.line(img, (x1+50, y1+50), (x2+50, y2+50), (0, 0, 0), 2)

    # Add scores
    cv2.putText(img, f"REBA: {reba_total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, f"RULA: {rula_total}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Add legend
    cv2.putText(img, "Legend:", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.rectangle(img, (200, 50), (220, 70), (0, 255, 0), -1)
    cv2.putText(img, "Low", (225, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.rectangle(img, (200, 80), (220, 100), (0, 255, 255), -1)
    cv2.putText(img, "Moderate", (225, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.rectangle(img, (200, 110), (220, 130), (0, 0, 255), -1)
    cv2.putText(img, "High", (225, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    cv2.imwrite(output_path, img)

# ----------- MAIN SCRIPT -----------
print("REBA Worksheet Additions:")
load_force_score = int(input("Enter load/force score (0=<5kg, 1=5-10kg, 2=>10kg): "))
activity_score = int(input("Enter activity score (0=static, 1=repeated/small, 2=rapid/unstable): "))

print("NIOSH Lifting Inputs:")
load_weight = float(input("Enter actual load weight (kg): "))
H = float(input("Enter horizontal distance (cm): "))
V = float(input("Enter vertical location (cm): "))
D = float(input("Enter vertical travel distance (cm): "))
F = float(input("Enter frequency (lifts/min): "))
A = float(input("Enter asymmetry angle (degrees): "))
C = input("Enter coupling quality (good/fair/poor): ").lower()

model = YOLO(model_path)
results = model.predict(source=image_path, conf=0.5, verbose=False)

if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
    keypoints = results[0].keypoints.data[0].cpu().numpy().flatten()
    reba_result = evaluate_reba_corrected(keypoints, load_force_score, activity_score)
    rula_result = evaluate_rula_corrected(keypoints)
    niosh_result = evaluate_niosh(load_weight, H, V, D, F, A, C)

    annotated_img = results[0].plot()
    cv2.imwrite(annotated_img_path, annotated_img)

    draw_dynamic_visualization(keypoints, reba_result['scores'], reba_result['total_score'], rula_result['total_score'], visualization_path)

    # mapping lowercase angle keys to capitalized score keys (REBA)
    angle_to_score_key = {
   	"trunk": "Trunk",
   	"neck": "Neck",
   	"upper_arm": "Upper Arm",
   	"lower_arm": "Lower Arm",
   	"wrist": "Wrist",
   	"leg": "Leg"
    }
    # mapping for RULA (no leg)
    rula_angle_to_score_key = {
   	"trunk": "Trunk",
   	"neck": "Neck",
   	"upper_arm": "Upper Arm",
   	"lower_arm": "Lower Arm",
   	"wrist": "Wrist"
    }
    # REBA DataFrame (keep same order as angles returned)
    reba_parts = list(reba_result["angles"].keys())
    reba_angles = [ round(reba_result["angles"][p], 2) for p in reba_parts ]
    reba_scores = [ reba_result["scores"].get(angle_to_score_key[p], None) for p in reba_parts ]
    df_reba = pd.DataFrame({
      "Body Part": reba_parts,
      "Angle (degrees)": reba_angles,
      "REBA Score": reba_scores
    })

    # RULA DataFrame — filter out any angles that RULA does not score (e.g. 'leg')
    rula_parts = [ p for p in rula_result["angles"].keys() if p in rula_angle_to_score_key ]
    rula_angles = [ round(rula_result["angles"][p], 2) for p in rula_parts ]
    rula_scores = [ rula_result["scores"].get(rula_angle_to_score_key[p], None) for p in rula_parts ]
    df_rula = pd.DataFrame({
      "Body Part": rula_parts,
      "Angle (degrees)": rula_angles,
      "RULA Score": rula_scores
     })

    df_niosh = pd.DataFrame({
    	"Metric": ["Recommended Weight Limit (RWL)", "Lifting Index (LI)", "Inference"],
    	"Value": [niosh_result["RWL"], niosh_result["LI"], niosh_result["inference"]]
    })

    summary_df = pd.DataFrame({
        "Summary": [
            "REBA Total Score", "REBA Inference",
            "RULA Total Score", "RULA Inference",
            "NIOSH RWL", "NIOSH LI", "NIOSH Inference"
        ],
        "Value": [
            reba_result["total_score"], reba_result["inference"],
            rula_result["total_score"], rula_result["inference"],
            niosh_result["RWL"], niosh_result["LI"], niosh_result["inference"]
        ]
    })

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        df_reba.to_excel(writer, sheet_name="REBA Breakdown", index=False)
        df_rula.to_excel(writer, sheet_name="RULA Breakdown", index=False)
        df_niosh.to_excel(writer, sheet_name="NIOSH Evaluation", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    # PDF Report
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 30), "Ergonomic Evaluation Report", fontsize=16)
    page.insert_image(fitz.Rect(50, 60, 250, 260), filename=annotated_img_path)
    page.insert_image(fitz.Rect(270, 60, 470, 260), filename=visualization_path)

    y = 280
    page.insert_text((50, y), f"REBA Score: {reba_result['total_score']} → {reba_result['inference']}", fontsize=12)
    page.insert_text((50, y+20), f"RULA Score: {rula_result['total_score']} → {rula_result['inference']}", fontsize=12)
    page.insert_text((50, y+40), f"NIOSH RWL: {niosh_result['RWL']} kg", fontsize=12)
    page.insert_text((50, y+60), f"NIOSH LI: {niosh_result['LI']} → {niosh_result['inference']}", fontsize=12)

    # REBA Breakdown Table
    y += 90
    page.insert_text((50, y), "REBA Breakdown:", fontsize=12)
    for i, row in df_reba.iterrows():
        page.insert_text((50, y+20+i*15), f"{row['Body Part']}: Angle={row['Angle (degrees)']}°, Score={row['REBA Score']}", fontsize=10)

    # RULA Breakdown Table
    y += 20 + len(df_reba)*15
    page.insert_text((50, y), "RULA Breakdown:", fontsize=12)
    for i, row in df_rula.iterrows():
        page.insert_text((50, y+20+i*15), f"{row['Body Part']}: Angle={row['Angle (degrees)']}°, Score={row['RULA Score']}", fontsize=10)

    doc.save(pdf_path)
    doc.close()

    print("All reports generated successfully.")
else:
    print("No person detected in the image.")