import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import cv2
import time

# ----------------------------
# Load YOLO model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ----------------------------
# UI
# ----------------------------
st.title("🎥 Live Object Detection & Tracking")
st.write("Turn on your camera to detect objects in real time.")

st.sidebar.header("⚙️ Settings")

save_frames = st.sidebar.checkbox("Save Detected Frames")
alert_object = st.sidebar.text_input("Alert Object (e.g., person)")

object_counts = {}

# ----------------------------
# Video Processing Class (more stable than callback mode)
# ----------------------------
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        global object_counts

        img = frame.to_ndarray(format="bgr24")

        CONFIDENCE = 0.5

        results = model.track(
            img,
            persist=True,
            conf=CONFIDENCE,
            verbose=False
        )

        annotated_frame = results[0].plot()

        object_counts = {}

        if results[0].boxes is not None:
            classes = results[0].boxes.cls.tolist()
            names = model.names

            for cls in classes:
                label = names[int(cls)]
                object_counts[label] = object_counts.get(label, 0) + 1

                # ALERT SYSTEM
                if alert_object and label == alert_object:
                    cv2.putText(
                        annotated_frame,
                        "ALERT!",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

        # SAVE FRAME (optional)
        if save_frames:
            filename = f"frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, annotated_frame)

        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ----------------------------
# WebRTC CONFIG (FIXED FOR STREAMLIT CLOUD)
# ----------------------------
st.write("📷 Camera Feed Below")

webrtc_streamer(
    key="object-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration={
        "iceServers": [
            {"urls": "stun:stun.l.google.com:19302"}
        ]
    }
)
