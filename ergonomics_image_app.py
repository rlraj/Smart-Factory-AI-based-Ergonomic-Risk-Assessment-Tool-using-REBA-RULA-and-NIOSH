import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
from ultralytics import YOLO
import fitz  # PyMuPDF

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

def process_image(image_path, model_path, params, output_folder):
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=0.5, verbose=False)
    annotated_img_path = os.path.join(output_folder, "annotated_image.jpg")
    excel_path = os.path.join(output_folder, "Ergonomic_Evaluation_Report.xlsx")
    pdf_path = os.path.join(output_folder, "Ergonomic_Evaluation_Report.pdf")
    if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
        keypoints = results[0].keypoints.data[0].cpu().numpy().flatten()
        reba_result = evaluate_reba(keypoints, params["load_force_score"], params["activity_score"])
        rula_result = evaluate_rula(keypoints)
        niosh_result = evaluate_niosh(params["load_weight"], params["H"], params["V"], params["D"], params["F"], params["A"], params["C"])
        annotated_img = results[0].plot()
        cv2.imwrite(annotated_img_path, annotated_img)
        # Excel output
        df_reba = pd.DataFrame({
            "Body Part": list(reba_result["angles"].keys()),
            "Angle (degrees)": [round(a, 2) for a in reba_result["angles"].values()],
            "REBA Score": [reba_result["scores"][part] for part in reba_result["angles"].keys()]
        })
        df_rula = pd.DataFrame({
            "Body Part": list(rula_result["angles"].keys()),
            "Angle (degrees)": [round(a, 2) for a in rula_result["angles"].values()],
            "RULA Score": [rula_result["scores"][part] for part in rula_result["angles"].keys()]
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
        # PDF output
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), "Ergonomic Evaluation Report", fontsize=16, fontname="helv", fill=(0, 0, 0))
        image_rect = fitz.Rect(50, 80, 300, 300)
        page.insert_image(image_rect, filename=annotated_img_path)
        text_y = 400
        page.insert_text((50, text_y), f"REBA Score: {reba_result['total_score']} → {reba_result['inference']}", fontsize=12)
        page.insert_text((50, text_y + 20), f"RULA Score: {rula_result['total_score']} → {rula_result['inference']}", fontsize=12)
        page.insert_text((50, text_y + 40), f"NIOSH RWL: {niosh_result['RWL']} kg", fontsize=12)
        page.insert_text((50, text_y + 60), f"NIOSH LI: {niosh_result['LI']} → {niosh_result['inference']}", fontsize=12)
        doc.save(pdf_path)
        doc.close()
        return annotated_img_path, excel_path, pdf_path
    else:
        return None, None, None

# ----------- STREAMLIT UI -----------

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
    submitted = st.form_submit_button("Run Evaluation")

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

uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
model_path = st.text_input("Enter YOLO model path", value="best.pt"
)
output_folder = st.text_input("Enter output folder", value="output")
os.makedirs(output_folder, exist_ok=True)

if uploaded_image and submitted:
    temp_image_path = os.path.join("temp", uploaded_image.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())
    annotated_img_path, excel_path, pdf_path = process_image(
        temp_image_path, model_path, params, output_folder
    )
    if annotated_img_path:
        st.image(annotated_img_path, caption="Annotated Image")
        st.download_button("Download Excel Report", open(excel_path, "rb"), file_name="Ergonomic_Evaluation_Report.xlsx")
        st.download_button("Download PDF Report", open(pdf_path, "rb"), file_name="Ergonomic_Evaluation_Report.pdf")
    else:
        st.error("No person detected in the image.")




