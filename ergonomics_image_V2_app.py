import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import fitz  # PyMuPDF

# --- Angle and Keypoint Utilities (from V6) ---
EPS = 1e-8

def angle_at_joint(p1, p2, p3):
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
    v = vector(p_top, p_bottom)
    vertical = np.array([0., 1.0]) if image_y_increases_down else np.array([0., -1.0])
    return angle_between_vectors(v, vertical)

def extract_angles_from_keypoints_flat(keypoints_flat):
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
    shoulder_mid = ((kp['left_shoulder'][0] + kp['right_shoulder'][0]) / 2.0,
                    (kp['left_shoulder'][1] + kp['right_shoulder'][1]) / 2.0)
    hip_mid = ((kp['left_hip'][0] + kp['right_hip'][0]) / 2.0,
               (kp['left_hip'][1] + kp['right_hip'][1]) / 2.0)
    trunk_ang = angle_to_vertical(shoulder_mid, hip_mid)
    head_point = kp.get('nose', shoulder_mid)
    neck_ang = angle_between_vectors(vector(head_point, shoulder_mid), vector(shoulder_mid, hip_mid))
    upper_arm_ang = angle_between_vectors(vector(shoulder_mid, kp['left_elbow']), vector(shoulder_mid, hip_mid))
    lower_arm_ang = angle_at_joint(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])
    wrist_ang = angle_between_vectors(vector(kp['left_elbow'], kp['left_wrist']), np.array([0., 1.0]))
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

def map_reba_buckets(angles):
    trunk_score = 1 if angles['trunk'] <= 10 else 2 if angles['trunk'] <= 20 else 3
    neck_score = 1 if angles['neck'] <= 10 else 2 if angles['neck'] <= 20 else 3
    leg_score = 1 if angles['leg'] <= 30 else 2 if angles['leg'] <= 60 else 3
    upper_arm_score = 1 if angles['upper_arm'] <= 20 else 2 if angles['upper_arm'] <= 60 else 3
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

def evaluate_reba_corrected(keypoints_flat, load_force_score=0, activity_score=0, coupling_score=1):
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

def process_image(image_path, model_path, params, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.5, verbose=False)
    annotated_img_path = os.path.join(output_folder, "annotated_image.jpg")
    visualization_path = os.path.join(output_folder, "ergonomic_visual.png")
    excel_path = os.path.join(output_folder, "Ergonomic_Evaluation_Report.xlsx")
    pdf_path = os.path.join(output_folder, "Ergonomic_Evaluation_Report.pdf")
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        keypoints = results[0].keypoints.data[0].cpu().numpy().flatten()
        reba_result = evaluate_reba_corrected(keypoints, params["load_force_score"], params["activity_score"])
        rula_result = evaluate_rula_corrected(keypoints)
        niosh_result = evaluate_niosh(params["load_weight"], params["H"], params["V"], params["D"], params["F"], params["A"], params["C"])
        annotated_img = results[0].plot()
        cv2.imwrite(annotated_img_path, annotated_img)
        draw_dynamic_visualization(keypoints, reba_result['scores'], reba_result['total_score'], rula_result['total_score'], visualization_path)
        # Excel Report
        df_reba = pd.DataFrame({
            "Body Part": list(reba_result["angles"].keys()),
            "Angle (degrees)": [round(a, 2) for a in reba_result["angles"].values()],
            "REBA Score": [reba_result["scores"][part.replace('_',' ').title()] for part in reba_result["angles"].keys()]
        })
        df_rula = pd.DataFrame({
            "Body Part": [p for p in rula_result["angles"].keys() if p in ["trunk","neck","upper_arm","lower_arm","wrist"]],
            "Angle (degrees)": [round(rula_result["angles"][p], 2) for p in rula_result["angles"].keys() if p in ["trunk","neck","upper_arm","lower_arm","wrist"]],
            "RULA Score": [rula_result["scores"][p.replace('_',' ').title()] for p in rula_result["angles"].keys() if p in ["trunk","neck","upper_arm","lower_arm","wrist"]]
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
        return annotated_img_path, visualization_path, excel_path, pdf_path, reba_result, rula_result, niosh_result
    else:
        return None

# --- Streamlit UI ---
st.title("Ergonomic Evaluation: REBA, RULA, NIOSH")

with st.form("params_form"):
    load_force_score = st.number_input("REBA Load/Force Score (0=<5kg, 1=5-10kg, 2=>10kg)", min_value=0, max_value=2, value=1)
    activity_score = st.number_input("REBA Activity Score (0=static, 1=repeated/small, 2=rapid/unstable)", min_value=0, max_value=2, value=1)
    load_weight = st.number_input("NIOSH Actual Load Weight (kg)", min_value=0.0, value=8.0)
    H = st.number_input("NIOSH Horizontal Distance (cm)", min_value=0.0, value=30.0)
    V = st.number_input("NIOSH Vertical Location (cm)", min_value=0.0, value=60.0)
    D = st.number_input("NIOSH Vertical Travel Distance (cm)", min_value=0.0, value=20.0)
    F = st.number_input("NIOSH Frequency (lifts/min)", min_value=0.0, value=1.0)
    A = st.number_input("NIOSH Asymmetry Angle (degrees)", min_value=0.0, value=45.0)
    C = st.selectbox("NIOSH Coupling Quality", ["good", "fair", "poor"])
    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    model_path = st.text_input("Enter YOLO model path", value="C:/Users/220250572/OneDrive - Regal Rexnord/Desktop/Ergonmics/runs/pose/train/weights/best.pt")
    output_folder = st.text_input("Enter output folder", value="output")
    submitted = st.form_submit_button("Run Evaluation")

if submitted:
    if uploaded_image:
        temp_image_path = os.path.join("temp", uploaded_image.name)
        os.makedirs("temp", exist_ok=True)
        with open(temp_image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        params = {
            "load_force_score": load_force_score,
            "activity_score": activity_score,
            "load_weight": load_weight,
            "H": H,
            "V": V,
            "D": D,
            "F": F,
            "A": A,
            "C": C
        }
        result = process_image(temp_image_path, model_path, params, output_folder)
        if result:
            annotated_img_path, visualization_path, excel_path, pdf_path, reba_result, rula_result, niosh_result = result
            st.image(annotated_img_path, caption="Annotated Pose")
            st.image(visualization_path, caption="Ergonomic Visualization")
            st.subheader("Summary")
            st.write(f"**REBA Score:** {reba_result['total_score']} → {reba_result['inference']}")
            st.write(f"**RULA Score:** {rula_result['total_score']} → {rula_result['inference']}")
            st.write(f"**NIOSH RWL:** {niosh_result['RWL']} kg")
            st.write(f"**NIOSH LI:** {niosh_result['LI']} → {niosh_result['inference']}")
            with open(excel_path, "rb") as f:
                st.download_button("Download Excel Report", f, file_name="Ergonomic_Evaluation_Report.xlsx")
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF Report", f, file_name="Ergonomic_Evaluation_Report.pdf")
        else:
            st.error("No person detected in the image.")
    else:
        st.warning("Please upload an image before running evaluation.")