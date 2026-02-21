import streamlit as st
import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel
import pytesseract
import re
import time
from PIL import Image
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Traffic Light Detector",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS for developer theme
st.markdown("""
<style>
    .main {
        background-color: #0a0c10;
    }
    .stApp {
        background-color: #0a0c10;
        color: #e6edf3;
    }
    .css-1d391kg {
        background-color: #0d1117;
    }
    h1, h2, h3 {
        color: #e6edf3 !important;
        font-family: 'SF Mono', Monaco, monospace !important;
    }
    .stButton button {
        background-color: #161b22;
        color: #e6edf3;
        border: 1px solid #2d333b;
        font-family: 'SF Mono', Monaco, monospace;
    }
    .stButton button:hover {
        border-color: #7bc96f;
    }
    .stButton button:active {
        background-color: #7bc96f;
    }
    .stAlert {
        background-color: #161b22;
        border: 1px solid #2d333b;
    }
    .stMarkdown {
        font-family: 'SF Mono', Monaco, monospace;
    }
    code {
        color: #7bc96f !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'camera_source' not in st.session_state:
    st.session_state.camera_source = 'webcam'  # 'webcam' or 'esp'
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'latest_detection' not in st.session_state:
    st.session_state.latest_detection = {
        'isTrafficLightDetected': False,
        'colorDetected': 'unknown',
        'isCountDownVisible': False,
        'countdown': None
    }

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

@st.cache_resource
def load_model():
    """Load YOLO model with caching (runs only once)"""
    safe_classes = [
        DetectionModel,
        nn.Module,
        nn.Sequential,
        nn.Conv2d,
        nn.BatchNorm2d,
        nn.SiLU,
        nn.ModuleList,
        dict,
        list
    ]
    
    with st.spinner("Loading YOLO model..."):
        try:
            with torch.serialization.safe_globals(safe_classes):
                model = YOLO('yolov8n.pt')
            st.success("✅ YOLO model loaded")
            return model
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return None

class TrafficLightDetector:
    def __init__(self, model):
        self.model = model
        self.target_class = 9
        
        # Color ranges in HSV
        self.color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255)],
            'red2': [(160, 100, 100), (180, 255, 255)],
            'yellow': [(20, 100, 100), (30, 255, 255)],
            'green': [(40, 100, 100), (80, 255, 255)]
        }
    
    def detect(self, frame):
        if frame is None:
            return st.session_state.latest_detection
        
        # Run YOLO inference
        results = self.model(frame, classes=[self.target_class], conf=0.3)
        
        output = {
            'isTrafficLightDetected': False,
            'colorDetected': 'unknown',
            'isCountDownVisible': False,
            'countdown': None
        }
        
        if len(results[0].boxes) == 0:
            return output
        
        # Get largest traffic light
        boxes = results[0].boxes.xyxy.cpu().numpy()
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = np.argmax(areas)
        x1, y1, x2, y2 = map(int, boxes[largest_idx])
        
        tl_roi = frame[y1:y2, x1:x2]
        if tl_roi.size == 0:
            return output
        
        output['isTrafficLightDetected'] = True
        
        # Detect color
        output['colorDetected'] = self._detect_color(tl_roi)
        
        # Detect countdown
        countdown_value, countdown_visible = self._detect_countdown(frame, (x1, y2, x2, y2 + 80))
        output['isCountDownVisible'] = countdown_visible
        output['countdown'] = countdown_value
        
        # Draw on frame
        self._draw_annotations(frame, (x1, y1, x2, y2), output['colorDetected'], countdown_value)
        
        return output
    
    def _detect_color(self, roi):
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        height = roi.shape[0]
        
        section_height = max(1, height // 3)
        sections = {
            'top': hsv[0:section_height, :],
            'middle': hsv[section_height:2*section_height, :] if height >= 2*section_height else hsv[section_height:height, :],
            'bottom': hsv[2*section_height:3*section_height, :] if height >= 3*section_height else hsv[2*section_height:height, :]
        }
        
        color_counts = {'red': 0, 'yellow': 0, 'green': 0}
        for sect_name, sect_hsv in sections.items():
            if sect_hsv.size == 0:
                continue
            for color_name, (lower, upper) in self.color_ranges.items():
                if color_name == 'red2':
                    continue
                mask = cv2.inRange(sect_hsv, np.array(lower), np.array(upper))
                if color_name == 'red':
                    mask2 = cv2.inRange(sect_hsv, np.array(self.color_ranges['red2'][0]), np.array(self.color_ranges['red2'][1]))
                    mask = cv2.bitwise_or(mask, mask2)
                count = cv2.countNonZero(mask)
                color_counts[color_name] += count
        
        total_pixels = sum(color_counts.values())
        if total_pixels < 50:
            return 'unknown'
        
        return max(color_counts, key=color_counts.get)
    
    def _detect_countdown(self, frame, countdown_region):
        x1, y1, x2, y2 = countdown_region
        h, w = frame.shape[:2]
        y1 = max(0, y1)
        y2 = min(h, y2)
        x1 = max(0, x1)
        x2 = min(w, x2)
        
        if y2 <= y1 or x2 <= x1:
            return None, False
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, False
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        
        try:
            text = pytesseract.image_to_string(thresh, config=custom_config).strip()
            if text.isdigit():
                return int(text), True
            numbers = re.findall(r'\d+', text)
            if numbers:
                return int(numbers[0]), True
        except:
            pass
        
        return None, False
    
    def _draw_annotations(self, frame, bbox, color, countdown):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        color_map = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'unknown': (255, 255, 255)
        }
        text_color = color_map.get(color, (255, 255, 255))
        cv2.putText(frame, f"Color: {color}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        if countdown is not None:
            cv2.putText(frame, f"Count: {countdown}s", (x1, y2+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

def init_camera(source):
    """Initialize camera based on source"""
    if source == 'webcam':
        return cv2.VideoCapture(0)
    else:
        esp_url = st.session_state.get('esp_url', "http://192.168.1.100:81/stream")
        return cv2.VideoCapture(esp_url)

# Sidebar
with st.sidebar:
    st.markdown("## 🚦 **traffic-light.detect()**")
    st.markdown("---")
    
    # Camera selection
    st.markdown("### Camera Source")
    camera_option = st.radio(
        "Select camera:",
        ["💻 Laptop Webcam", "📡 ESP32-CAM"],
        index=0 if st.session_state.camera_source == 'webcam' else 1
    )
    
    new_source = 'webcam' if camera_option == "💻 Laptop Webcam" else 'esp'
    
    if new_source != st.session_state.camera_source:
        st.session_state.camera_source = new_source
        if st.session_state.camera:
            st.session_state.camera.release()
        st.session_state.camera = init_camera(new_source)
        st.rerun()
    
    # ESP32 URL input (if ESP selected)
    if st.session_state.camera_source == 'esp':
        esp_url = st.text_input(
            "ESP32-CAM Stream URL:",
            value=st.session_state.get('esp_url', "http://192.168.1.100:81/stream")
        )
        st.session_state.esp_url = esp_url
    
    st.markdown("---")
    
    # Control buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.running = True
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.running = False
            if st.session_state.camera:
                st.session_state.camera.release()
                st.session_state.camera = None
    
    st.markdown("---")
    
    # Status
    st.markdown("### System Status")
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.markdown("**Camera:**")
        st.markdown("**Model:**")
        st.markdown("**Tesseract:**")
    with status_col2:
        cam_status = "✅ Ready" if st.session_state.camera and st.session_state.camera.isOpened() else "❌ Offline"
        st.markdown(f"`{cam_status}`")
        st.markdown("`✅ Loaded`")
        tess_ok = "✅ OK" if pytesseract.get_tesseract_version() else "❌ Missing"
        st.markdown(f"`{tess_ok}`")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## # live-feed")
    
    # Video feed placeholder
    video_placeholder = st.empty()
    
    # Load model
    if st.session_state.detector is None:
        model = load_model()
        if model:
            st.session_state.detector = TrafficLightDetector(model)
    
    # Initialize camera if needed
    if st.session_state.camera is None:
        st.session_state.camera = init_camera(st.session_state.camera_source)
    
    # Video processing loop
    if st.session_state.running and st.session_state.camera and st.session_state.camera.isOpened():
        frame_count = 0
        while st.session_state.running:
            ret, frame = st.session_state.camera.read()
            if not ret:
                st.warning("Failed to grab frame")
                break
            
            # Run detection every 3rd frame
            if frame_count % 3 == 0:
                detection = st.session_state.detector.detect(frame)
                st.session_state.latest_detection = detection
            
            # Convert BGR to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            frame_count += 1
            time.sleep(0.03)  # ~30 fps
    else:
        # Show placeholder when not running
        placeholder = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera Stopped", (200, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        video_placeholder.image(placeholder, channels="RGB", use_container_width=True)

with col2:
    st.markdown("## # detection.result")
    st.markdown(f"`polling: real-time`")
    
    # Detection results
    detection = st.session_state.latest_detection
    
    # Traffic Light Status
    with st.container():
        st.markdown("### 🚦 `isTrafficLightDetected()`")
        if detection['isTrafficLightDetected']:
            st.markdown("**✅ `true`**")
        else:
            st.markdown("**❌ `false`**")
    
    # Color Detected
    with st.container():
        st.markdown("### 🎨 `getColor()`")
        color = detection['colorDetected']
        color_display = {
            'red': '🔴 red',
            'yellow': '🟡 yellow',
            'green': '🟢 green',
            'unknown': '⚪ unknown'
        }.get(color, '⚪ unknown')
        st.markdown(f"**`{color_display}`**")
    
    # Countdown Visible
    with st.container():
        st.markdown("### ⏱️ `isCountdownVisible()`")
        if detection['isCountDownVisible']:
            st.markdown("**✅ `true`**")
        else:
            st.markdown("**❌ `false`**")
    
    # Countdown Value
    with st.container():
        st.markdown("### 🔢 `getCountdown()`")
        if detection['countdown'] is not None:
            st.markdown(f"**`{detection['countdown']}s`**")
        else:
            st.markdown("**`null`**")
    
    st.markdown("---")
    
    # API Endpoint info (if you still want to expose it)
    st.markdown("### 📡 API Endpoint")
    st.code("GET http://localhost:8501/api/detection", language="bash")
    
    # Export data button
    if st.button("📋 Copy Detection JSON"):
        import json
        st.json(detection)

# Cleanup on exit
import atexit
def cleanup():
    if st.session_state.camera:
        st.session_state.camera.release()

atexit.register(cleanup)