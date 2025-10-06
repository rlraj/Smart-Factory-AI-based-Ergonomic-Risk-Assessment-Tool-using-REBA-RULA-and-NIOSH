import streamlit as st

import numpy as np

import pandas as pd

import cv2

import os

import io

import zipfile

import tempfile

from ultralytics import YOLO

# =========================

# --------- MATH ----------

# =========================

def calculate_angle(p1, p2, p3):

    a = np.array(p1, dtype=float)

    b = np.array(p2, dtype=float)

    c = np.array(p3, dtype=float)

    ba = a - b

    bc = c - b

    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6

    cosine_angle = np.dot(ba, bc) / denom

    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return float(np.degrees(angle))

def evaluate_reba(keypoints, load_force_score, activity_score):

    kp_names = [

        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',

        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',

        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',

        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'

    ]

    kp = {name: (keypoints[i*3], keypoints[i*3+1]) for i, name in enumerate(kp_names)}

    trunk_angle = calculate_angle(kp['left_shoulder'], kp['left_hip'],   kp['left_knee'])

    neck_angle  = calculate_angle(kp['nose'],         kp['left_shoulder'], kp['left_hip'])

    leg_angle   = calculate_angle(kp['left_hip'],     kp['left_knee'],   kp['left_ankle'])

    upper_arm_angle = calculate_angle(kp['left_shoulder'], kp['left_elbow'], kp['left_wrist'])

    lower_arm_angle = calculate_angle(kp['left_elbow'],    kp['left_wrist'], kp['left_hip'])

    wrist_angle     = calculate_angle(kp['left_elbow'],    kp['left_wrist'], kp['left_hip'])

    coupling_score = 1

    trunk_score      = 1 if trunk_angle < 20 else 2 if trunk_angle < 60 else 3

    neck_score       = 1 if neck_angle  < 20 else 2 if neck_angle  < 45 else 3

    leg_score        = 1 if leg_angle   < 30 else 2 if leg_angle   < 60 else 3

    upper_arm_score  = 1 if upper_arm_angle < 45 else 2 if upper_arm_angle < 90 else 3

    lower_arm_score  = 1 if lower_arm_angle < 60 else 2 if lower_arm_angle < 100 else 3

    wrist_score      = 1 if wrist_angle < 15 else 2 if wrist_angle < 30 else 3

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

    lower_arm_angle = calculate_angle(kp['left_elbow'],    kp['left_wrist'], kp['left_hip'])

    wrist_angle     = calculate_angle(kp['left_elbow'],    kp['left_wrist'], kp['left_hip'])

    neck_angle      = calculate_angle(kp['nose'],          kp['left_shoulder'], kp['left_hip'])

    trunk_angle     = calculate_angle(kp['left_shoulder'], kp['left_hip'], kp['left_knee'])

    upper_arm_score = 1 if upper_arm_angle < 45 else 2 if upper_arm_angle < 90 else 3

    lower_arm_score = 1 if lower_arm_angle < 60 else 2 if lower_arm_angle < 100 else 3

    wrist_score     = 1 if wrist_angle < 15 else 2 if wrist_angle < 30 else 3

    neck_score      = 1 if neck_angle < 20 else 2 if neck_angle < 45 else 3

    trunk_score     = 1 if trunk_angle < 20 else 2 if trunk_angle < 60 else 3

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

    LC = 23.0

    HM = 25.0 / H if H > 0 else 0.0

    VM = 1.0 - 0.003 * abs(V - 75.0)

    DM = 0.82 + 4.5 / D if D > 0 else 0.0

    AM = 1.0 - 0.0032 * A

    FM = 0.94 if F < 0.2 else 0.88 if F < 0.5 else 0.75

    CM = 1.0 if C == "good" else 0.95 if C == "fair" else 0.9

    RWL = LC * HM * VM * DM * AM * FM * CM

    LI = (load_weight / RWL) if RWL > 0 else 0.0

    inference = "Safe lifting task." if LI <= 1 else "Unsafe lifting task. Ergonomic improvements needed."

    return {"RWL": round(RWL, 2), "LI": round(LI, 2), "inference": inference}

# =========================

# ------ VIDEO I/O --------

# =========================

def make_videowriter(path_mp4, fps, size_wh):

    w, h = size_wh

    # Prefer H.264 for HTML5 playback

    fourcc_avc1 = cv2.VideoWriter_fourcc(*'avc1')

    vw = cv2.VideoWriter(path_mp4, fourcc_avc1, fps, (w, h))

    if not vw.isOpened():

        # Fallback: MP4V

        fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')

        vw = cv2.VideoWriter(path_mp4, fourcc_mp4v, fps, (w, h))

    if not vw.isOpened():

        raise RuntimeError("Failed to open VideoWriter with avc1/mp4v")

    return vw

def process_video(video_path, model_path, params, output_video_path, excel_path):

    """

    Returns: (output_video_path, excel_path, dfs_dict)

    """

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    # Metadata

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)  or 0)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    fps    = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 1e-3:

        fps = 25  # safe fallback

    # Read first frame to ensure size

    ret, first_frame = cap.read()

    if not ret:

        cap.release()

        raise RuntimeError("Could not read frames from the video.")

    if width == 0 or height == 0:

        height, width = first_frame.shape[:2]

    out = make_videowriter(output_video_path, fps, (width, height))

    reba_rows, rula_rows, niosh_rows, summary_rows = [], [], [], []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    progress = st.progress(0, text="Processing video…")

    frame_count = 0

    def handle_frame(img):

        nonlocal frame_count

        frame_count += 1

        results = model(img, conf=0.5, verbose=False)

        if results and results[0].keypoints is not None and len(results[0].keypoints.data) > 0:

            keypoints = results[0].keypoints.data[0].cpu().numpy().flatten()

            reba_result  = evaluate_reba(keypoints, params["load_force_score"], params["activity_score"])

            rula_result  = evaluate_rula(keypoints)

            niosh_result = evaluate_niosh(params["load_weight"], params["H"], params["V"],

                                          params["D"], params["F"], params["A"], params["C"])

            annotated_img = results[0].plot()  # BGR

            out.write(annotated_img)

            for part in reba_result["angles"]:

                reba_rows.append({

                    "Frame": frame_count,

                    "Body Part": part,

                    "Angle (degrees)": round(reba_result["angles"][part], 2),

                    "REBA Score": reba_result["scores"][part]

                })

            for part in rula_result["angles"]:

                rula_rows.append({

                    "Frame": frame_count,

                    "Body Part": part,

                    "Angle (degrees)": round(rula_result["angles"][part], 2),

                    "RULA Score": rula_result["scores"][part]

                })

            niosh_rows.append({

                "Frame": frame_count,

                "RWL": niosh_result["RWL"],

                "LI": niosh_result["LI"],

                "Inference": niosh_result["inference"]

            })

            summary_rows.append({

                "Frame": frame_count,

                "REBA Total Score": reba_result["total_score"],

                "REBA Inference": reba_result["inference"],

                "RULA Total Score": rula_result["total_score"],

                "RULA Inference": rula_result["inference"],

                "NIOSH RWL": niosh_result["RWL"],

                "NIOSH LI": niosh_result["LI"],

                "NIOSH Inference": niosh_result["inference"]

            })

        else:

            # keep duration consistent even if no keypoints

            out.write(img)

        if total_frames:

            progress.progress(min(frame_count / max(total_frames, 1), 1.0),

                              text=f"Processing frame {frame_count}/{total_frames}")

    # process first + rest

    handle_frame(first_frame)

    while True:

        ret, frame = cap.read()

        if not ret:

            break

        handle_frame(frame)

    cap.release()

    out.release()

    progress.empty()

    # DataFrames

    df_reba    = pd.DataFrame(reba_rows)

    df_rula    = pd.DataFrame(rula_rows)

    df_niosh   = pd.DataFrame(niosh_rows)

    df_summary = pd.DataFrame(summary_rows)

    # Excel export

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        df_reba.to_excel(writer,    sheet_name="REBA Breakdown", index=False)

        df_rula.to_excel(writer,    sheet_name="RULA Breakdown", index=False)

        df_niosh.to_excel(writer,   sheet_name="NIOSH Evaluation", index=False)

        df_summary.to_excel(writer, sheet_name="Summary",         index=False)

    dfs = {

        "REBA Breakdown": df_reba,

        "RULA Breakdown": df_rula,

        "NIOSH Evaluation": df_niosh,

        "Summary": df_summary

    }

    return output_video_path, excel_path, dfs

# =========================

# -------- UI -------------

# =========================

st.title("Ergonomic Evaluation from Video (Annotated Pose + Excel)")

with st.form("params_form"):

    load_force_score = st.number_input("REBA Load/Force Score (0=<5kg, 1=5-10kg, 2=>10kg)",

                                       min_value=0, max_value=2, value=1)

    activity_score   = st.number_input("REBA Activity Score (0=static, 1=repeated/small, 2=rapid/unstable)",

                                       min_value=0, max_value=2, value=1)

    load_weight = st.number_input("NIOSH Actual Load Weight (kg)", min_value=0.0, value=8.0)

    H = st.number_input("NIOSH Horizontal Distance (cm)", min_value=0.0, value=30.0)

    V = st.number_input("NIOSH Vertical Location (cm)", min_value=0.0, value=60.0)

    D = st.number_input("NIOSH Vertical Travel Distance (cm)", min_value=0.0, value=20.0)

    F = st.number_input("NIOSH Frequency (lifts/min)", min_value=0.0, value=1.0)

    A = st.number_input("NIOSH Asymmetry Angle (degrees)", min_value=0.0, value=45.0)

    C = st.selectbox("NIOSH Coupling Quality", ["good", "fair", "poor"])

    model_path = st.text_input(

        "Enter YOLO model path",

        value="C:/Users/220250572/OneDrive - Regal Rexnord/Desktop/Ergonmics/runs/pose/train/weights/best.pt"

    )

    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mpeg4"])

    submitted = st.form_submit_button("Run Evaluation")

params = {

    "load_force_score": load_force_score,

    "activity_score": activity_score,

    "load_weight": load_weight,

    "H": H, "V": V, "D": D, "F": F, "A": A, "C": C

}

if uploaded_video and submitted:

    # Save uploaded file to temp

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_video.name)[1]) as temp_video:

        temp_video.write(uploaded_video.read())

        temp_video_path = temp_video.name

    # Output paths

    output_video_path = os.path.join(tempfile.gettempdir(), "annotated_pose_video.mp4")

    excel_path = os.path.join(tempfile.gettempdir(), "Ergonomic_Evaluation_Breakdown.xlsx")

    try:

        video_file, excel_file, dfs = process_video(

            temp_video_path, model_path, params, output_video_path, excel_path

        )

    except Exception as e:

        st.error(f"Processing failed: {e}")

        st.stop()

    # Preview player

    st.subheader("Annotated Video")

    st.video(video_file)

    # Downloads

    st.subheader("Downloads")

    # Excel button

    with open(excel_file, "rb") as f:

        st.download_button(

            "Download Excel Report (All Sheets)",

            data=f,

            file_name="Ergonomic_Evaluation_Breakdown.xlsx",

            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

        )

    # Video button

    with open(video_file, "rb") as f:

        st.download_button(

            "Download Annotated Pose Video (MP4)",

            data=f,

            file_name="annotated_pose.mp4",

            mime="video/mp4"

        )

    # One-click ZIP containing both

    buf = io.BytesIO()

    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:

        z.write(excel_file, arcname="Ergonomic_Evaluation_Breakdown.xlsx")

        z.write(video_file, arcname="annotated_pose.mp4")

    buf.seek(0)

    st.download_button(

        "Download Both (ZIP)",

        data=buf,

        file_name="ergonomics_outputs.zip",

        mime="application/zip"

    )

    # Quick summary preview

    st.subheader("Summary (Preview)")

    st.dataframe(dfs["Summary"].head(20))
