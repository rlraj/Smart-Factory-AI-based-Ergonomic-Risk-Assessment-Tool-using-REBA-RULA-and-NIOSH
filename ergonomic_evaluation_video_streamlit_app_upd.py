import streamlit as st

import numpy as np

import pandas as pd

import cv2

import os

import io

import zipfile

import tempfile

from ultralytics import YOLO

import plotly.express as px

import fitz  # PyMuPDF

# =========================

# --------- MATH ----------

# =========================

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

# =========================

# -------- UI -------------

# =========================

st.title("Ergonomic Video Evaluation & Automated PDF Report")

with st.form("params_form"):
    load_force_score = st.number_input("REBA Load/Force Score (0=<5kg, 1=5-10kg, 2=>10kg)", min_value=0, max_value=2, value=1)
    activity_score   = st.number_input("REBA Activity Score (0=static, 1=repeated/small, 2=rapid/unstable)", min_value=0, max_value=2, value=1)
    load_weight = st.number_input("NIOSH Actual Load Weight (kg)", min_value=0.0, value=8.0)
    H = st.number_input("NIOSH Horizontal Distance (cm)", min_value=0.0, value=30.0)
    V = st.number_input("NIOSH Vertical Location (cm)", min_value=0.0, value=60.0)
    D = st.number_input("NIOSH Vertical Travel Distance (cm)", min_value=0.0, value=20.0)
    F = st.number_input("NIOSH Frequency (lifts/min)", min_value=0.0, value=1.0)
    A = st.number_input("NIOSH Asymmetry Angle (degrees)", min_value=0.0, value=45.0)
    C = st.selectbox("NIOSH Coupling Quality", ["good", "fair", "poor"])
    model_path = st.text_input("YOLO model path", value="best.pt")
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mpeg4"])
    submitted = st.form_submit_button("Run Evaluation")

params = {
    "load_force_score": load_force_score,
    "activity_score": activity_score,
    "load_weight": load_weight,
    "H": H, "V": V, "D": D, "F": F, "A": A, "C": C
}

if uploaded_video and submitted:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as temp_video:
        temp_video.write(uploaded_video.read())
        temp_video_path = temp_video.name

    output_folder = tempfile.gettempdir()
    output_video_path = os.path.join(output_folder, "annotated_pose_video.mp4")
    excel_path = os.path.join(output_folder, "Ergonomic_Evaluation_Breakdown.xlsx")
    pdf_path = os.path.join(output_folder, "Ergonomic_Evaluation_Report.pdf")

    # ----------- Video Analysis -----------
    model = YOLO(model_path)
    cap = cv2.VideoCapture(temp_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    frame_results = []
    frame_number = 0
    progress = st.progress(0, text="Processing video…")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number += 1
        results = model.predict(source=frame, conf=0.5, verbose=False)
        if results[0].keypoints is not None and len(results[0].keypoints.data) > 0:
            keypoints = results[0].keypoints.data[0].cpu().numpy().flatten()
            reba_result = evaluate_reba_corrected(keypoints, load_force_score, activity_score)
            rula_result = evaluate_rula_corrected(keypoints)
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
                "RULA Score": rula_result["total_score"],
                "RULA Inference": rula_result["inference"],
                "NIOSH RWL": niosh_result["RWL"],
                "NIOSH LI": niosh_result["LI"],
                "NIOSH Inference": niosh_result["inference"],
                "Keypoints": keypoints.tolist()
            })
        else:
            out.write(frame)
        if total_frames:
            progress.progress(min(frame_number / max(total_frames, 1), 1.0), text=f"Processing frame {frame_number}/{total_frames}")

    cap.release()
    out.release()
    progress.empty()

    # ----------- Excel Export -----------
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

    # ----------- PDF Report Generation -----------
    joint_columns = ["Trunk Score", "Neck Score", "Leg Score", "Upper Arm Score", "Lower Arm Score", "Wrist Score"]
    joint_summary = []
    for joint in joint_columns:
        avg = df_video[joint].mean()
        min_score = df_video[joint].min()
        max_score = df_video[joint].max()
        high_risk_frames = df_video[df_video[joint] == 3].shape[0]
        joint_summary.append({
            "Joint": joint.replace(" Score", ""),
            "Average Score": round(avg, 2),
            "Min Score": min_score,
            "Max Score": max_score,
            "High Risk Frames": high_risk_frames
        })
    joint_summary_df = pd.DataFrame(joint_summary)

    fig = px.bar(joint_summary_df, x="Joint", y="Average Score",
                 title="Average Ergonomic Risk Score per Joint",
                 labels={"Average Score": "Avg Score"}, text="Average Score")
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis=dict(range=[0, 3.5]))
    chart_path = os.path.join(output_folder, "joint_risk_bar_chart.png")
    

    skeleton_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    color_map = {1: (0, 255, 0), 2: (0, 255, 255), 3: (0, 0, 255)}
    joint_map = {
        "Trunk": [5, 11],
        "Neck": [0, 5],
        "Leg": [11, 13, 15],
        "Upper Arm": [5, 7],
        "Lower Arm": [7, 9],
        "Wrist": [9]
    }

    image_paths = []
    for joint in joint_map:
        col_name = joint + " Score"
        if df_video[df_video[col_name] == 3].shape[0] > 0:
            frame_row = df_video[df_video[col_name] == 3].iloc[0]
        else:
            max_score = df_video[col_name].max()
            frame_row = df_video[df_video[col_name] == max_score].iloc[0]
        keypoints = frame_row["Keypoints"]
        img = np.ones((800, 800, 3), dtype=np.uint8) * 255
        for p1, p2 in skeleton_pairs:
            x1, y1 = int(keypoints[p1*3]), int(keypoints[p1*3+1])
            x2, y2 = int(keypoints[p2*3]), int(keypoints[p2*3+1])
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
        for i in range(17):
            x, y = int(keypoints[i*3]), int(keypoints[i*3+1])
            color = (0, 0, 0)
            for j_idx in joint_map[joint]:
                if i == j_idx:
                    color = color_map[frame_row[col_name]]
            cv2.circle(img, (x, y), 6, color, -1)
        cv2.putText(img, f"Joint: {joint}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        cv2.putText(img, f"Frame: {int(frame_row['Frame'])}", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        pose_path = os.path.join(output_folder, f"pose_{joint}_frame_{int(frame_row['Frame'])}.png")
        cv2.imwrite(pose_path, img)
        image_paths.append((joint, pose_path))

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 30), "Ergonomic Evaluation Summary", fontsize=14)
    for i, row in summary_df.iterrows():
        page.insert_text((50, 60 + i*20), f"{row['Metric']}: {row['Value']}", fontsize=10)

    page = doc.new_page()
    page.insert_text((50, 30), "Joint-wise Ergonomic Risk Summary", fontsize=14)
    for i, row in joint_summary_df.iterrows():
        page.insert_text((50, 60 + i*20), f"{row['Joint']}: Avg={row['Average Score']}, Min={row['Min Score']}, Max={row['Max Score']}, High Risk Frames={row['High Risk Frames']}", fontsize=10)

    page.insert_image(fitz.Rect(50, 200, 550, 500), filename=chart_path)

    for joint, img_path in image_paths:
        page = doc.new_page()
        page.insert_text((50, 30), f"Pose Skeleton for {joint} (High Risk Frame)", fontsize=14)
        page.insert_image(fitz.Rect(50, 60, 550, 860), filename=img_path)

    doc.save(pdf_path)
    doc.close()

    # ----------- Streamlit Outputs -----------
    st.success("Processing complete!")
    st.subheader("Downloads")
    with open(excel_path, "rb") as f:
        st.download_button("Download Excel Report", f, file_name="Ergonomic_Evaluation_Breakdown.xlsx")
    with open(pdf_path, "rb") as f:
        st.download_button("Download PDF Report", f, file_name="Ergonomic_Evaluation_Report.pdf")
    with open(output_video_path, "rb") as f:
        st.download_button("Download Annotated Video", f, file_name="annotated_pose_video.mp4")

    # One-click ZIP download
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.write(excel_path, arcname="Ergonomic_Evaluation_Report.xlsx")
        z.write(pdf_path, arcname="Ergonomic_Evaluation_Report.pdf")
        z.write(output_video_path, arcname="annotated_pose_video.mp4")
    buf.seek(0)
    st.download_button(
        "Download All Outputs (ZIP)",
        data=buf,
        file_name="ergonomics_outputs.zip",
        mime="application/zip"
    )

    st.subheader("Summary (Preview)")
    st.dataframe(summary_df)

    st.subheader("Joint Risk Bar Chart")
    st.plotly_chart(fig, use_container_width=True, key="joint_risk_chart")

    st.subheader("Pose Skeletons for High Risk Frames")
    for joint, img_path in image_paths:

        st.image(img_path, caption=f"{joint} (High Risk Frame)")

