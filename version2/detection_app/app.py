import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from utils.detection_utils import DetectionUtils
from utils.gym_utils import GymUtils

# Page config
st.set_page_config(page_title="YOLOv8 Deployment Suite", layout="wide")
st.title("🚀 YOLOv8 Computer Vision Suite")
st.markdown("Object Detection | Counting | Segmentation | Pose | Customer Detection | AI Workout")

# Sidebar for mode selection
st.sidebar.header("Configuration")
mode = st.sidebar.selectbox(
    "Select Task",
    ["Object Detection", "Object Counting", "Segmentation", "Pose Estimation",
     "Customer Detection", "AI Workout Monitor"]
)

# Confidence slider
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 0.9, 0.45, 0.05)

# Source selection
source_type = st.sidebar.radio("Input Source", ["Upload Image", "Upload Video", "Webcam"])

# Video processing speed optimization
if source_type == "Upload Video":
    process_every_n_frames = st.sidebar.slider("Process every N frames (1 = all frames, higher = faster)", 1, 10, 1)
    st.sidebar.info("Higher values skip more frames for faster processing, but may miss detections.")

# Model initialization (cached)
@st.cache_resource
def load_detection_utils():
    return DetectionUtils()

@st.cache_resource
def load_gym_utils(exercise):
    return GymUtils(exercise_type=exercise)

detection_utils = load_detection_utils()

# AI Workout specific options
if mode == "AI Workout Monitor":
    exercise = st.sidebar.selectbox("Exercise Type", ["pushup", "squat", "pullup", "benchpress", "bicepcurl", "overheadpress"])
    gym_utils = load_gym_utils(exercise)

# Function to process image
def process_image(frame):
    if mode == "Object Detection":
        _, annotated = detection_utils.detect_objects(frame, conf_threshold)
        return annotated
    elif mode == "Object Counting":
        _, annotated = detection_utils.count_objects(frame, conf_threshold)
        return annotated
    elif mode == "Segmentation":
        return detection_utils.segment_objects(frame, conf_threshold)
    elif mode == "Pose Estimation":
        return detection_utils.pose_estimate(frame, conf_threshold)
    elif mode == "Customer Detection":
        return detection_utils.customer_detection(frame, conf_threshold)
    elif mode == "AI Workout Monitor":
        return gym_utils.process_frame(frame)
    else:
        return frame

# Function to process video with frame skipping
def process_video(video_path, output_path, process_every_n=1):
    cap = cv2.VideoCapture(video_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    progress_bar = st.progress(0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0
    frame_counter = 0
    last_processed_frame = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process only every Nth frame
        if frame_counter % process_every_n == 0:
            processed = process_image(frame)
            last_processed_frame = processed
        else:
            # Use last processed frame or original
            processed = last_processed_frame if last_processed_frame is not None else frame
        
        out.write(processed)
        processed_frames += 1
        frame_counter += 1
        progress_bar.progress(processed_frames / frame_count)

    cap.release()
    out.release()
    return output_path

# Main area
if source_type == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(frame, caption="Original", use_container_width=True)

        with st.spinner("Processing..."):
            result = process_image(frame)
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB) if isinstance(result, np.ndarray) else result

        with col2:
            st.image(result_rgb, caption="Processed", use_container_width=True)

        # Download button
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        is_success, buffer = cv2.imencode(".jpg", result_bgr)
        if is_success:
            st.download_button("Download Result", data=buffer.tobytes(), file_name="output.jpg", mime="image/jpeg")

elif source_type == "Upload Video":
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name

        if st.button("Process Video"):
            with st.spinner("Processing video... This may take a while."):
                result_path = process_video(video_path, output_path, process_every_n=process_every_n_frames)
            st.success("Processing complete!")
            st.video(result_path)

            # Download button
            with open(result_path, "rb") as f:
                st.download_button("Download Processed Video", data=f, file_name="output.mp4", mime="video/mp4")

elif source_type == "Webcam":
    st.warning("⚠️ Webcam mode will open your camera. Make sure you've granted permissions.")
    
    # Webcam control buttons
    col1, col2 = st.columns(2)
    with col1:
        run_webcam = st.button("Start Webcam")
    with col2:
        stop_webcam = st.button("Stop Webcam")
    
    # Frame placeholder
    stframe = st.empty()
    
    if run_webcam:
        cap = cv2.VideoCapture(0)
        # Set lower resolution for faster processing
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = process_image(frame_rgb)
            stframe.image(result, channels="RGB", use_container_width=True)
            
            if stop_webcam:
                break
        
        cap.release()
        st.rerun()

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using YOLOv8, OpenCV, and Streamlit")