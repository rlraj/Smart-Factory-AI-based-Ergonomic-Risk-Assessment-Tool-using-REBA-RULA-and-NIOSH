from ultralytics import YOLO
import numpy as np
import cv2
import os
import pandas as pd
import fitz  # PyMuPDF

model_path = "C:/Users/220250572/OneDrive - Regal Rexnord/Desktop/Ergonmics/runs/pose/train/weights/best.pt"
output_folder = r"C:/Users/220250572/OneDrive - Regal Rexnord/Desktop/Ergonmics/output/"
os.makedirs(output_folder, exist_ok=True)
video_save_path = os.path.join(output_folder, "webcam_reba_analysis.mp4")
excel_path = os.path.join(output_folder, "REBA_Evaluation_Webcam_Report.xlsx")
pdf_path = os.path.join(output_folder, "REBA_Evaluation_Webcam_Report.pdf")

def calculate_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def evaluate_reba(keypoints):
    kp_names = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    kp = {name: (keypoints[i*3], keypoints[i*3+1]) for i, name in enumerate(kp_names)}
    def safe_angle(a, b, c):
        try:
            return calculate_angle(a, b, c)
        except Exception:
            return None
    trunk_angle = safe_angle(kp['left_shoulder'], kp['left_hip'], kp['left_knee'])
    neck_angle = safe_angle(kp['nose'], kp['left_shoulder'], kp['left_hip'])
    leg_angle = safe_angle(kp['left_hip'], kp['left_knee'], kp['left_ankle'])
    upper_arm_angle = safe_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])
    lower_arm_angle = safe_angle(kp['left_elbow'], kp['left_wrist'], kp['left_hip'])
    wrist_angle = safe_angle(kp['left_elbow'], kp['left_wrist'], kp['left_hip'])
    coupling_score = 1

    trunk_score = 1 if trunk_angle is not None and trunk_angle < 20 else 2 if trunk_angle is not None and trunk_angle < 60 else 3 if trunk_angle is not None else None
    neck_score = 1 if neck_angle is not None and neck_angle < 20 else 2 if neck_angle is not None and neck_angle < 45 else 3 if neck_angle is not None else None
    leg_score = 1 if leg_angle is not None and leg_angle < 30 else 2 if leg_angle is not None and leg_angle < 60 else 3 if leg_angle is not None else None
    upper_arm_score = 1 if upper_arm_angle is not None and upper_arm_angle < 45 else 2 if upper_arm_angle is not None and upper_arm_angle < 90 else 3 if upper_arm_angle is not None else None
    lower_arm_score = 1 if lower_arm_angle is not None and lower_arm_angle < 60 else 2 if lower_arm_angle is not None and lower_arm_angle < 100 else 3 if lower_arm_angle is not None else None
    wrist_score = 1 if wrist_angle is not None and wrist_angle < 15 else 2 if wrist_angle is not None and wrist_angle < 30 else 3 if wrist_angle is not None else None

    scores = {
        "Trunk": trunk_score,
        "Neck": neck_score,
        "Leg": leg_score,
        "Upper Arm": upper_arm_score,
        "Lower Arm": lower_arm_score,
        "Wrist": wrist_score,
        "Coupling": coupling_score
    }
    reba_score = sum([s for s in scores.values() if s is not None])
    risk_areas = [part for part, score in scores.items() if score == 3]
    occluded_areas = [part for part, score in scores.items() if score is None]

    if reba_score <= 3:
        inference = "Ergonomically safe (Low risk)"
    elif reba_score <= 7:
        inference = "Moderate ergonomic risk. Consider improvements."
    elif reba_score <= 10:
        inference = "High ergonomic risk. Improvements recommended soon."
    else:
        inference = "Very high ergonomic risk. Immediate action required."

    return reba_score, inference, risk_areas, occluded_areas

model = YOLO(model_path)
cap = cv2.VideoCapture(0)
fps = 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
recording = False

frame_data = []

print("Press 's' to start recording, 'e' to end/save, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.predict(source=frame, conf=0.5, verbose=False)
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        keypoints = results[0].keypoints.data[0].cpu().numpy().flatten()
        reba_score, inference, risk_areas, occluded_areas = evaluate_reba(keypoints)
        annotated_img = results[0].plot()
        y_offset = 30
        cv2.putText(annotated_img, f"REBA: {reba_score} ({inference})", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        y_offset += 30
        if risk_areas:
            cv2.putText(annotated_img, f"High risk: {', '.join(risk_areas)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            y_offset += 25
        if occluded_areas:
            cv2.putText(annotated_img, f"Occluded: {', '.join(occluded_areas)}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
        if recording:
            frame_data.append({
                "REBA Score": reba_score,
                "Inference": inference,
                "High Risk Areas": ", ".join(risk_areas),
                "Occluded Areas": ", ".join(occluded_areas)
            })
            out.write(annotated_img)
    else:
        annotated_img = frame
        cv2.putText(annotated_img, "No person detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        if recording:
            frame_data.append({
                "REBA Score": None,
                "Inference": "No person detected",
                "High Risk Areas": "",
                "Occluded Areas": ""
            })
            out.write(annotated_img)

    cv2.imshow("YOLOv8 Pose REBA Risk", annotated_img)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        if not recording:
            out = cv2.VideoWriter(video_save_path, fourcc, fps, (width, height))
            recording = True
            print("Recording started...")
    if key == ord('e'):
        if recording:
            recording = False
            out.release()
            print(f"Recording ended. Video saved to {video_save_path}")
            # Generate Excel report
            df = pd.DataFrame(frame_data)
            summary = {
                "Average REBA Score": df["REBA Score"].dropna().mean(),
                "Max REBA Score": df["REBA Score"].dropna().max(),
                "Min REBA Score": df["REBA Score"].dropna().min(),
                "Most Common Inference": df["Inference"].mode()[0] if not df["Inference"].mode().empty else "N/A"
            }
            summary_df = pd.DataFrame(list(summary.items()), columns=["Metric", "Value"])
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Frame Analysis", index=False)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)

            # Generate PDF report
            doc = fitz.open()
            page = doc.new_page()
            page.insert_text((50, 50), "REBA Webcam Evaluation Report", fontsize=16, fontname="helv", fill=(0, 0, 0))
            y = 80
            for metric, value in summary.items():
                page.insert_text((50, y), f"{metric}: {round(value, 2) if isinstance(value, (int, float)) else value}", fontsize=12)
                y += 20
            page.insert_text((50, y + 20), "See Excel for frame-by-frame details.", fontsize=12)
            doc.save(pdf_path)
            doc.close()
            print(f"Excel and PDF reports saved to {output_folder}")
    if key == ord('q'):
        break

cap.release()
if out is not None and recording:
    out.release()
cv2.destroyAllWindows()

