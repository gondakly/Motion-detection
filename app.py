import cv2
import numpy as np
import time
from collections import defaultdict, deque
import streamlit as st
from ultralytics import YOLO
import tempfile
import os

# ========================= CONFIGURATION =========================
RESIZE_W = 800
RESIZE_H = 600

st.set_page_config(page_title="Smart Traffic Surveillance", layout="wide", page_icon="🚦")
st.title("Smart YOLO Traffic Surveillance System")
st.markdown("**Real-time Vehicle Detection, Tracking & Counting**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("Control Panel")
   
    uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov", "mkv"])
   
    confidence = st.slider("Confidence Threshold (%)", 25, 95, 45)
    line_pos = st.slider("Counting Line Position (%)", 15, 85, 55)
    vel_threshold = st.slider("Velocity Threshold", 0.5, 5.0, 1.8, 0.1)
    trail_length = st.slider("Trail Length", 8, 30, 12)
    box_thickness = st.slider("Box Thickness", 1, 6, 2)
    frame_skip = st.slider("Frame Skip (Higher = Faster)", 0, 6, 1)
   
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        start_btn = st.button("Start Processing", type="primary", use_container_width=True)
    with col2:
        stop_btn = st.button("Stop", use_container_width=True)
    with col3:
        reset_btn = st.button("Reset Counters", use_container_width=True)

# ====================== LAYOUT ======================
col_video, col_stats = st.columns([3, 1])

with col_video:
    st.subheader("Live Video")
    video_frame = st.empty()
    motion_frame = st.empty()

with col_stats:
    st.subheader("Statistics")
    in_count = st.metric("IN (↓)", 0)
    out_count = st.metric("OUT (↑)", 0)
    total_cars_st = st.metric("Total Cars", 0)
    active_st = st.metric("Active Objects", 0)
    fps_st = st.metric("FPS", 0)

# ====================== MODEL ======================
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ====================== SESSION STATE ======================
if 'running' not in st.session_state:
    st.session_state.running = False
    st.session_state.count_in = 0
    st.session_state.count_out = 0
    st.session_state.total_cars = 0
    st.session_state.tracked_objects = {}
    st.session_state.object_id_counter = 0

# ====================== BUTTONS ======================
if start_btn:
    st.session_state.running = True

if stop_btn:
    st.session_state.running = False

if reset_btn:
    st.session_state.count_in = 0
    st.session_state.count_out = 0
    st.session_state.total_cars = 0
    st.session_state.tracked_objects.clear()
    st.session_state.object_id_counter = 0
    st.rerun()

# ====================== MAIN PROCESSING ======================
if st.session_state.running:
    # Video Source
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.getvalue())
        video_source = tfile.name
    else:
        video_source = r"D:/Final Project Computer Vision/Traffic Motion.mp4"

    cap = cv2.VideoCapture(video_source)
    back_sub = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=40, detectShadows=False)

    tracked_objects = st.session_state.tracked_objects
    object_id_counter = st.session_state.object_id_counter

    frame_count = 0
    prev_time = time.time()
    fps = 0

    while cap.isOpened() and st.session_state.running:
        ret, frame = cap.read()
        if not ret:
            st.warning("Video has ended.")
            st.session_state.running = False
            break

        # Frame Skip
        for _ in range(frame_skip):
            cap.read()

        frame = cv2.resize(frame, (RESIZE_W, RESIZE_H))
        H, W = frame.shape[:2]

        # Motion Detection
        fgmask = back_sub.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 180, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=2)
        fgmask = cv2.dilate(fgmask, kernel, iterations=3)

        # YOLO
        results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                            conf=confidence/100.0, iou=0.45, verbose=False)

        current_ids = set()
        LINE_Y = int(H * line_pos / 100)
        VELOCITY_THRESHOLD = vel_threshold

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, conf, cls_id, track_id in zip(boxes, confs, class_ids, track_ids):
                x1, y1, x2, y2 = map(int, box)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                class_name = model.names[int(cls_id)]
                label = {"car": "Car", "truck": "Car", "bus": "Car",
                         "bicycle": "Bicycle", "motorcycle": "Bicycle",
                         "person": "Person"}.get(class_name, class_name.capitalize())

                current_ids.add(track_id)

                if track_id not in tracked_objects:
                    object_id_counter += 1
                    tracked_objects[track_id] = {
                        "centroid": (cx, cy),
                        "velocity": (0, 0),
                        "class": label,
                        "counted": False,
                        "history": deque(maxlen=trail_length)
                    }

                obj = tracked_objects[track_id]
                px, py = obj["centroid"]
                vx, vy = cx - px, cy - py

                obj["velocity"] = (vx*0.72 + obj["velocity"][0]*0.28,
                                  vy*0.72 + obj["velocity"][1]*0.28)
                obj["centroid"] = (cx, cy)
                obj["history"].append((cx, cy))

                if not obj["counted"]:
                    vel_y = obj["velocity"][1]
                    if py < LINE_Y <= cy and vel_y > VELOCITY_THRESHOLD:
                        st.session_state.count_in += 1
                        obj["counted"] = True
                        if label == "Car": st.session_state.total_cars += 1
                    elif py > LINE_Y >= cy and vel_y < -VELOCITY_THRESHOLD:
                        st.session_state.count_out += 1
                        obj["counted"] = True
                        if label == "Car": st.session_state.total_cars += 1

                color = (0, 180, 255) if label == "Car" else (255, 0, 255) if label == "Bicycle" else (50, 220, 50)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                cv2.putText(frame, f"{label} #{track_id}", (x1, max(y1-8,15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                for i in range(1, len(obj["history"])):
                    if obj["history"][i-1] and obj["history"][i]:
                        cv2.line(frame, obj["history"][i-1], obj["history"][i], color, 2)

        # FPS
        frame_count += 1
        if time.time() - prev_time >= 1.0:
            fps = frame_count
            frame_count = 0
            prev_time = time.time()

        cv2.line(frame, (0, LINE_Y), (W, LINE_Y), (0, 255, 255), 3)

        # Display
        video_frame.image(frame, channels="BGR", use_column_width=True)
        motion_frame.image(fgmask, channels="GRAY", caption="Motion Mask (MOG2)", use_column_width=True)

        # Update Stats
        in_count.metric("IN (↓)", st.session_state.count_in)
        out_count.metric("OUT (↑)", st.session_state.count_out)
        total_cars_st.metric("Total Cars", st.session_state.total_cars)
        active_st.metric("Active Objects", len(tracked_objects))
        fps_st.metric("FPS", round(fps, 1))

        # Small delay to prevent Streamlit media cache crash
        time.sleep(0.01)

    cap.release()
    if uploaded_file is not None and 'tfile' in locals():
        try:
            os.unlink(tfile.name)
        except:
            pass

else:
    st.info("Upload a video (optional) then click Start Processing")

st.caption("Built with YOLOv8 • OpenCV • Streamlit")
