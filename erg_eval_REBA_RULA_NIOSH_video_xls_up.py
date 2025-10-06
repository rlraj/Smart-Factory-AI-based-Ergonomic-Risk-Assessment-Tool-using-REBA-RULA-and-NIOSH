from ultralytics import YOLO
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import cv2
import os

# ----------- CONFIGURATION -----------
video_path = r"C:/Users/220250572/Desktop/Ergonmics/test_videos/test_video2.mp4" 
output_folder = r"C:/Users/220250572/Desktop/Ergonmics/output/videos"
os.makedirs(output_folder, exist_ok=True)
model_path = "C:/Users/220250572/Desktop/Ergonmics/runs/pose/train/weights/best.pt"  
excel_path = os.path.join(output_folder, "test_video2_Report.xlsx")
pdf_path = os.path.join(output_folder, "test_video2_Report.pdf")
annotated_video_path = os.path.join(output_folder, "test_video2_pose_skeleton.mp4")

# ----------- FUNCTIONS -----------
def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def evaluate_reba(keypoints, load_force_score, activity_score):
    kp_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    kp = {name: (keypoints[i*3], keypoints[i*3+1]) for i, name in enumerate(kp_names)}
    trunk_angle = calculate_angle(kp['left_shoulder'], kp['left_hip'], kp['left_knee'])
    neck_angle = calculate_angle(kp['nose'], kp['left_shoulder'], kp['left_hip'])
    leg_angle = calculate_angle(kp['left_hip'], kp['left_knee'], kp['left_ankle'])
    upper_arm_angle = calculate_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])
    lower_arm_angle = calculate_angle(kp['left_elbow'], kp['left_wrist'], kp['left_hip'])
    wrist_angle = calculate_angle(kp['left_elbow'], kp['left_wrist'], kp['left_hip'])
    coupling_score = 1
    trunk_score = 1 if trunk_angle < 20 else 2 if trunk_angle < 60 else 3
    neck_score = 1 if neck_angle < 20 else 2 if neck_angle < 45 else 3
    leg_score = 1 if leg_angle < 30 else 2 if leg_angle < 60 else 3
    upper_arm_score = 1 if upper_arm_angle < 45 else 2 if upper_arm_angle < 90 else 3
    lower_arm_score = 1 if lower_arm_angle < 60 else 2 if lower_arm_angle < 100 else 3
    wrist_score = 1 if wrist_angle < 15 else 2 if wrist_angle < 30 else 3
    reba_score = (
        trunk_score + neck_score + leg_score +
        upper_arm_score + lower_arm_score + wrist_score +
        coupling_score + load_force_score + activity_score
    )
    if reba_score <= 3:
        inference = "Posture is ergonomically safe (Low risk)."
    elif reba_score <= 7:
        inference = "Posture shows moderate ergonomic risk. Consider improvements."
    elif reba_score <= 10:
        inference = "Posture shows high ergonomic risk. Improvements recommended soon."
    else:
        inference = "Posture shows very high ergonomic risk. Immediate action required."
    return {
        "angles": {
            "Trunk": trunk_angle,
            "Neck": neck_angle,
            "Leg": leg_angle,
            "Upper Arm": upper_arm_angle,
            "Lower Arm": lower_arm_angle,
            "Wrist": wrist_angle
        },
        "scores": {
            "Trunk": trunk_score,
            "Neck": neck_score,
            "Leg": leg_score,
            "Upper Arm": upper_arm_score,
            "Lower Arm": lower_arm_score,
            "Wrist": wrist_score,
            "Coupling": coupling_score,
            "Load/Force": load_force_score,
            "Activity": activity_score
        },
        "total_score": reba_score,
        "inference": inference
    }

def evaluate_rula(keypoints):
    kp_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    kp = {name: (keypoints[i*3], keypoints[i*3+1]) for i, name in enumerate(kp_names)}
    upper_arm_angle = calculate_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])
    lower_arm_angle = calculate_angle(kp['left_elbow'], kp['left_wrist'], kp['left_hip'])
    wrist_angle = calculate_angle(kp['left_elbow'], kp['left_wrist'], kp['left_hip'])
    neck_angle = calculate_angle(kp['nose'], kp['left_shoulder'], kp['left_hip'])
    trunk_angle = calculate_angle(kp['left_shoulder'], kp['left_hip'], kp['left_knee'])
    upper_arm_score = 1 if upper_arm_angle < 45 else 2 if upper_arm_angle < 90 else 3
    lower_arm_score = 1 if lower_arm_angle < 60 else 2 if lower_arm_angle < 100 else 3
    wrist_score = 1 if wrist_angle < 15 else 2 if wrist_angle < 30 else 3
    neck_score = 1 if neck_angle < 20 else 2 if neck_angle < 45 else 3
    trunk_score = 1 if trunk_angle < 20 else 2 if trunk_angle < 60 else 3
    rula_score = upper_arm_score + lower_arm_score + wrist_score + neck_score + trunk_score
    if rula_score <= 3:
        inference = "Acceptable posture."
    elif rula_score <= 5:
        inference = "Posture needs further investigation."
    elif rula_score <= 7:
        inference = "Posture needs changes soon."
    else:
        inference = "Posture needs immediate changes."
    return {
        "angles": {
            "Upper Arm": upper_arm_angle,
            "Lower Arm": lower_arm_angle,
            "Wrist": wrist_angle,
            "Neck": neck_angle,
            "Trunk": trunk_angle
        },
        "scores": {
            "Upper Arm": upper_arm_score,
            "Lower Arm": lower_arm_score,
            "Wrist": wrist_score,
            "Neck": neck_score,
            "Trunk": trunk_score
        },
        "total_score": rula_score,
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

# ----------- MAIN SCRIPT -----------
load_force_score = int(input("Enter REBA load/force score (0=<5kg, 1=5-10kg, 2=>10kg): "))
activity_score = int(input("Enter REBA activity score (0=static, 1=repeated/small, 2=rapid/unstable): "))
load_weight = float(input("Enter actual load weight (kg): "))
H = float(input("Enter horizontal distance (cm): "))
V = float(input("Enter vertical location (cm): "))
D = float(input("Enter vertical travel distance (cm): "))
F = float(input("Enter frequency (lifts/min): "))
A = float(input("Enter asymmetry angle (degrees): "))
C = input("Enter coupling quality (good/fair/poor): ").lower()

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
out = cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_results = []
frame_number = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_number += 1
    results = model.predict(source=frame, conf=0.5, verbose=False)
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        keypoints = results[0].keypoints.data[0].cpu().numpy().flatten()
        reba_result = evaluate_reba(keypoints, load_force_score, activity_score)
        rula_result = evaluate_rula(keypoints)
        niosh_result = evaluate_niosh(load_weight, H, V, D, F, A, C)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)
        frame_results.append({
            "Frame": frame_number,
            "REBA Score": reba_result["total_score"],
            "REBA Inference": reba_result["inference"],
            "Trunk Score": reba_result["scores"]["Trunk"],
            "Neck Score": reba_result["scores"]["Neck"],
            "Leg Score": reba_result["scores"]["Leg"],
            "Upper Arm Score": reba_result["scores"]["Upper Arm"],
            "Lower Arm Score": reba_result["scores"]["Lower Arm"],
            "Wrist Score": reba_result["scores"]["Wrist"],
            "Coupling Score": reba_result["scores"]["Coupling"],
            "Load/Force Score": reba_result["scores"]["Load/Force"],
            "Activity Score": reba_result["scores"]["Activity"],
            "RULA Score": rula_result["total_score"],
            "RULA Inference": rula_result["inference"],
            "NIOSH RWL": niosh_result["RWL"],
            "NIOSH LI": niosh_result["LI"],
            "NIOSH Inference": niosh_result["inference"]
        })
    else:
        out.write(frame)

cap.release()
out.release()

df_video = pd.DataFrame(frame_results)
summary_df = pd.DataFrame({
    "Metric": [
        "Average REBA Score", "Average RULA Score", "NIOSH RWL", "NIOSH LI", "NIOSH Inference"
    ],
    "Value": [
        round(df_video["REBA Score"].mean(), 2) if not df_video.empty else "N/A",
        round(df_video["RULA Score"].mean(), 2) if not df_video.empty else "N/A",
        frame_results[0]["NIOSH RWL"] if frame_results else "N/A",
        frame_results[0]["NIOSH LI"] if frame_results else "N/A",
        frame_results[0]["NIOSH Inference"] if frame_results else "N/A"
    ]
})

with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
    df_video.to_excel(writer, sheet_name="Framewise Breakdown", index=False)
    summary_df.to_excel(writer, sheet_name="Summary", index=False)

doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), "Ergonomic Evaluation Report", fontsize=16, fontname="helv", fill=(0, 0, 0))

# Add summary stats
page.insert_text((50, 80), f"Average REBA Score: {summary_df.iloc[0,1]}", fontsize=12)
page.insert_text((50, 100), f"Average RULA Score: {summary_df.iloc[1,1]}", fontsize=12)
page.insert_text((50, 120), f"NIOSH RWL: {summary_df.iloc[2,1]} kg", fontsize=12)
page.insert_text((50, 140), f"Lifting Index: {summary_df.iloc[3,1]}", fontsize=12)
page.insert_text((50, 160), f"NIOSH Inference: {summary_df.iloc[4,1]}", fontsize=12)
doc.save(pdf_path)
doc.close()

print(f"PDF report saved at: {pdf_path}")
print(f"Excel report saved at: {excel_path}")

print(f"Annotated video saved at: {annotated_video_path}")
