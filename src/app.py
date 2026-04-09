# import streamlit as st
# import numpy as np
# import cv2
# from PIL import Image
# import os
# import tempfile

# from ultralytics import YOLO

# from mqi import compute_mqi_from_mask
# from repair import repair_mask_in_memory
# from severity import compute_severity_from_mask
# from video_pipeline import process_video


# # -------------------------------------------------
# # PATH SETUP
# # -------------------------------------------------

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(BASE_DIR)

# MODEL_PATH = os.path.join(
#     PROJECT_ROOT,
#     "runs",
#     "segment",
#     "yolov8n_train",
#     "weights",
#     "best.pt"
# )


# # -------------------------------------------------
# # LOAD MODEL
# # -------------------------------------------------

# @st.cache_resource
# def load_model():
#     return YOLO(MODEL_PATH)

# model = load_model()


# # -------------------------------------------------
# # PAGE CONFIG
# # -------------------------------------------------

# st.set_page_config(page_title="PolypFusionNet", layout="wide")

# st.title("PolypFusionNet")

# st.subheader(
#     "Polyp Detection, Segmentation, MQI, Mask Repair, Severity Scoring and Tracking"
# )


# # -------------------------------------------------
# # FILE UPLOAD
# # -------------------------------------------------

# uploaded_file = st.file_uploader(
#     "Upload Colonoscopy Image or Video",
#     type=["jpg", "jpeg", "png", "mp4", "avi"]
# )

# if uploaded_file is None:
#     st.stop()

# file_type = uploaded_file.type


# # =================================================
# # IMAGE PIPELINE
# # =================================================

# if "image" in file_type:

#     image = Image.open(uploaded_file)
#     image_np = np.array(image)

#     # Handle RGBA images
#     if image_np.shape[-1] == 4:
#         image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
#     else:
#         image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

#     col1, col2 = st.columns(2)

#     with col1:
#         st.subheader("Original Image")
#         st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), width="stretch")

#     if st.button("Run PolypFusionNet"):

#         results = model.predict(
#             source=image_np,
#             conf=0.10,
#             imgsz=640
#         )[0]

#         annotated_frame = results.plot()

#         with col2:
#             st.subheader("Detection Result")
#             st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), width="stretch")

#         # Detection check
#         if results.boxes is None or len(results.boxes) == 0:
#             st.error("No polyp detected")
#             st.stop()

#         st.success(f"{len(results.boxes)} polyp(s) detected")

#         if results.masks is None:
#             st.warning("Polyp detected but segmentation mask unavailable")
#             st.stop()

#         # Use first detected mask
#         mask = results.masks.data[0].cpu().numpy()
#         binary_mask = (mask > 0.5).astype(np.uint8) * 255

#         st.subheader("Segmentation Mask")
#         st.image(binary_mask, width="stretch")

#         # MQI
#         mqi_score, mask_quality = compute_mqi_from_mask(binary_mask)

#         st.subheader("Mask Quality")

#         st.write("MQI Score:", round(mqi_score, 4))
#         st.write("Status:", mask_quality)

#         repaired_mask = binary_mask

#         if mask_quality == "Needs_Repair":

#             repaired_mask = repair_mask_in_memory(binary_mask)

#             st.subheader("Repaired Mask")
#             st.image(repaired_mask, width="stretch")

#         # Severity
#         severity_score, severity_label = compute_severity_from_mask(repaired_mask)

#         st.subheader("Severity Analysis")

#         st.write("Severity Score:", round(severity_score, 4))
#         st.write("Severity:", severity_label)


# # =================================================
# # VIDEO PIPELINE
# # =================================================

# elif "video" in file_type:

#     st.subheader("Video Analysis")

#     temp_video = tempfile.NamedTemporaryFile(delete=False)
#     temp_video.write(uploaded_file.read())

#     video_path = temp_video.name

#     cap = cv2.VideoCapture(video_path)
#     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     progress_bar = st.progress(0)
#     status_text = st.empty()

#     stframe = st.empty()

#     current_frame = 0
#     final_severity = None

#     for frame, severity in process_video(video_path, model):

#         current_frame += 1

#         progress = current_frame / total_frames
#         progress_bar.progress(progress)

#         status_text.text(
#             f"Processing frame {current_frame} / {total_frames}"
#         )

#         if severity is not None:
#             final_severity = severity

#         # Render fewer frames for performance
#         if current_frame % 25 == 0:

#             frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             stframe.image(frame_rgb, width="stretch")

#     cap.release()

#     progress_bar.empty()

#     status_text.text("Video processing completed")

#     st.success("Video analysis finished")

#     st.subheader("Final Severity Prediction")

#     if final_severity is not None:
#         st.write("Detected Severity:", final_severity)
#     else:
#         st.write("No polyp detected in video.")




















import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import tempfile

from ultralytics import YOLO

from mqi import compute_mqi_from_mask
from repair import repair_mask_in_memory
from severity import compute_severity_from_mask
from video_pipeline import process_video


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="PolypFusionNet",
    page_icon="🧬",
    layout="wide"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f4f8fb;
}

h1 {
    color: #0b3c5d;
    text-align: center;
}

h2, h3 {
    color: #1f4e79;
}

/* Metric Cards */

.metric-card {
    background-color: white;
    padding: 18px;
    border-radius: 10px;
    border-left: 6px solid #1f77b4;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.12);
    color: black;
    margin-bottom: 15px;
}

/* Success */

.success-box {
    background-color: #d4edda;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #28a745;
    color: #155724;
    font-size: 18px;
}

/* Error */

.error-box {
    background-color: #f8d7da;
    padding: 15px;
    border-radius: 10px;
    border-left: 6px solid #dc3545;
    color: #721c24;
}

/* RUN BUTTON */

div.stButton > button:first-child {
    background-color: #1f77b4;
    color: white;
    font-size: 18px;
    height: 3em;
    width: 100%;
    border-radius: 10px;
    border: none;
}

div.stButton > button:hover {
    background-color: #145a86;
    color: white;
}

</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# PATH SETUP
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "runs",
    "segment",
    "yolov8n_train",
    "weights",
    "best.pt"
)


# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()


# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

st.sidebar.title("PolypFusionNet")

st.sidebar.markdown("""
### AI Colonoscopy Analysis

Features:

• Polyp Detection  
• Segmentation  
• Mask Quality Index (MQI)  
• Mask Repair  
• Severity Scoring  
• Video Tracking
""")

st.sidebar.info("Upload an image or video to start analysis.")


# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("PolypFusionNet")

st.markdown(
"### AI Framework for Polyp Detection, Segmentation, MQI Analysis, Mask Repair and Severity Prediction"
)


# -------------------------------------------------
# OVERLAY FUNCTION
# -------------------------------------------------

def overlay_mask(image, mask, color=(0,255,0), alpha=0.4):

    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))

    overlay = image.copy()

    colored_mask = np.zeros_like(image)

    colored_mask[mask_resized > 0] = color

    overlay = cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0)

    return overlay


# -------------------------------------------------
# FILE UPLOAD
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Colonoscopy Image or Video",
    type=["jpg","jpeg","png","mp4","avi"]
)

if uploaded_file is None:
    st.stop()

file_type = uploaded_file.type


# =================================================
# IMAGE PIPELINE
# =================================================

if "image" in file_type:

    image = Image.open(uploaded_file)
    image_np = np.array(image)

    if image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
    else:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original Image")
        st.image(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB), use_container_width=True)

    if st.button("Run PolypFusionNet"):

        results = model.predict(
            source=image_np,
            conf=0.10,
            imgsz=640
        )[0]

        annotated_frame = results.plot()

        with col2:
            st.subheader("Detection Result")
            st.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        if results.boxes is None or len(results.boxes) == 0:

            st.markdown("""
            <div class="error-box">
            No Polyp Detected
            </div>
            """, unsafe_allow_html=True)

            st.stop()

        st.markdown(f"""
        <div class="success-box">
        {len(results.boxes)} Polyp(s) Detected Successfully
        </div>
        """, unsafe_allow_html=True)

        if results.masks is None:
            st.warning("Segmentation mask unavailable")
            st.stop()

        mask = results.masks.data[0].cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8) * 255

        with col3:
            st.subheader("Segmentation Mask")
            st.image(binary_mask, use_container_width=True)

        # MQI

        mqi_score, mask_quality = compute_mqi_from_mask(binary_mask)

        st.markdown(f"""
        <div class="metric-card">
        <h3>Mask Quality Index</h3>
        <p><b>MQI Score:</b> {round(mqi_score,4)}</p>
        <p><b>Status:</b> {mask_quality}</p>
        </div>
        """, unsafe_allow_html=True)

        repaired_mask = binary_mask

        if mask_quality == "Needs_Repair":

            repaired_mask = repair_mask_in_memory(binary_mask)

            st.subheader("Mask Repair Result")

            colA, colB = st.columns(2)

            with colA:
                st.image(repaired_mask, use_container_width=True)

            with colB:
                overlay = overlay_mask(image_np, repaired_mask)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), use_container_width=True)

        if mask_quality != "Reject":

            severity_score, severity_label = compute_severity_from_mask(repaired_mask)

            st.markdown(f"""
            <div class="metric-card">
            <h3>Severity Assessment</h3>
            <p><b>Severity Score:</b> {round(severity_score,4)}</p>
            <p><b>Severity Level:</b> {severity_label}</p>
            </div>
            """, unsafe_allow_html=True)

        else:

            st.markdown("""
            <div class="error-box">
            Mask Rejected — Severity Cannot Be Evaluated
            </div>
            """, unsafe_allow_html=True)


# =================================================
# VIDEO PIPELINE
# =================================================

elif "video" in file_type:

    st.subheader("Video Analysis Dashboard")

    video_col1, video_col2 = st.columns([3,1])

    with video_col1:
        st.subheader("Live Colonoscopy Analysis")
        stframe = st.empty()

    with video_col2:

        st.subheader("Frame Analysis")
        frame_panel = st.empty()

        st.subheader("Final Result")
        final_panel = st.empty()

    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_file.read())

    video_path = temp_video.name

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    progress_bar = st.progress(0)

    current_frame = 0
    final_severity = None

    for frame, severity, mqi_score, mask_quality, severity_label in process_video(video_path, model):

        current_frame += 1

        progress_bar.progress(current_frame / total_frames)

        severity_display = severity_label if severity_label else "Not Evaluated"

        frame_panel.markdown(f"""
        <div class="metric-card">
        <h3>Frame Analysis</h3>

        <p><b>Current Frame:</b> {current_frame}</p>
        <p><b>Total Frames:</b> {total_frames}</p>

        <hr>

        <p><b>MQI Score:</b> {mqi_score if mqi_score else "N/A"}</p>
        <p><b>Mask Quality:</b> {mask_quality if mask_quality else "N/A"}</p>

        <hr>

        <p><b>Severity:</b> {severity_display}</p>

        </div>
        """, unsafe_allow_html=True)

        if severity is not None:
            final_severity = severity

        if current_frame % 25 == 0:

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            stframe.image(frame_rgb, use_container_width=True)

    cap.release()

    st.success("Video analysis completed")

    final_panel.markdown(f"""
    <div class="metric-card">
    <h3>Final Severity Result</h3>
    <p><b>Severity Level:</b> {final_severity}</p>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.markdown("---")

st.markdown(
"<center>PolypFusionNet • AI Assisted Colonoscopy Analysis</center>",
unsafe_allow_html=True
)