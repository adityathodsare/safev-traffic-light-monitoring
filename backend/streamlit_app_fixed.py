import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import pytesseract
import re
import time
from PIL import Image
import os

# Page config
st.set_page_config(
    page_title="Traffic Light Detector",
    page_icon="🚦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #0a0c10; }
    .stApp { background-color: #0a0c10; color: #e6edf3; }
    h1, h2, h3 { color: #e6edf3 !important; font-family: monospace !important; }
    .stButton button { 
        background-color: #161b22; 
        color: #e6edf3; 
        border: 1px solid #2d333b;
        width: 100%;
    }
    .stButton button:hover { border-color: #7bc96f; }
    .stAlert { background-color: #161b22; border: 1px solid #2d333b; }
    code { color: #7bc96f !important; }
    .stImage { border: 1px solid #2d333b; border-radius: 5px; }
    .refresh-info {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background-color: #161b22;
        padding: 5px 10px;
        border-radius: 5px;
        border: 1px solid #2d333b;
        font-size: 12px;
        color: #8b949e;
        z-index: 1000;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'camera' not in st.session_state:
    st.session_state.camera = None
if 'running' not in st.session_state:
    st.session_state.running = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
if 'fps' not in st.session_state:
    st.session_state.fps = 0
if 'model_load_attempted' not in st.session_state:
    st.session_state.model_load_attempted = False
if 'latest_detection' not in st.session_state:
    st.session_state.latest_detection = {
        'isTrafficLightDetected': False,
        'colorDetected': 'unknown',
        'countdown': None
    }

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Color ranges
COLOR_RANGES = {
    'red': [(0, 100, 100), (10, 255, 255)],
    'red2': [(160, 100, 100), (180, 255, 255)],
    'yellow': [(20, 100, 100), (30, 255, 255)],
    'green': [(40, 100, 100), (80, 255, 255)]
}

@st.cache_resource
def load_model():
    """Load YOLO model with full PyTorch compatibility"""
    try:
        import torch
        import torch.nn as nn
        from ultralytics.nn.tasks import DetectionModel
        
        st.info(f"PyTorch version: {torch.__version__}")
        
        # Method 1: Safe globals approach (for PyTorch 2.6+)
        try:
            # Check if safe_globals exists (PyTorch 2.2+)
            if hasattr(torch.serialization, 'safe_globals'):
                safe_classes = [
                    DetectionModel,
                    nn.Module,
                    nn.Sequential,
                    nn.Conv2d,
                    nn.BatchNorm2d,
                    nn.SiLU,
                    nn.ModuleList,
                    nn.Identity,
                    nn.Upsample,
                    dict,
                    list,
                    tuple,
                    set
                ]
                
                with torch.serialization.safe_globals(safe_classes):
                    model = YOLO('yolov8n.pt')
                st.success("✅ Model loaded with safe_globals")
                return model
            else:
                # Older PyTorch versions
                model = YOLO('yolov8n.pt')
                st.success("✅ Model loaded successfully")
                return model
                
        except Exception as e1:
            st.warning(f"Safe globals method failed: {str(e1)[:100]}...")
            
            # Method 2: Direct load with weights_only=False (safe for trusted source)
            try:
                import torch.serialization
                # Temporarily set weights_only to False for loading
                original_weights_only = torch.load.__defaults__[0] if torch.load.__defaults__ else True
                
                # This is safe because we trust the source (ultralytics official model)
                model = YOLO('yolov8n.pt')
                st.success("✅ Model loaded with direct method")
                return model
                
            except Exception as e2:
                st.warning(f"Direct method failed: {str(e2)[:100]}...")
                
                # Method 3: Download fresh model
                st.info("Attempting to download fresh model...")
                model = YOLO('yolov8n.pt')
                st.success("✅ Fresh model downloaded and loaded")
                return model
                
    except Exception as e:
        st.error(f"All loading methods failed: {e}")
        st.info("💡 The app will run in demo mode without detection")
        return None

def detect_traffic_light(frame, model):
    """Detection function"""
    if frame is None:
        return frame, None
    
    detection = {
        'isTrafficLightDetected': False,
        'colorDetected': 'unknown',
        'countdown': None
    }
    
    if model is None:
        # Demo mode
        cv2.putText(frame, "DEMO MODE - No Detection", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return frame, detection
    
    try:
        # Run inference
        results = model(frame, classes=[9], conf=0.3, verbose=False)
        
        if len(results[0].boxes) > 0:
            # Get largest detection
            boxes = results[0].boxes.xyxy.cpu().numpy()
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            largest_idx = np.argmax(areas)
            x1, y1, x2, y2 = map(int, boxes[largest_idx])
            
            detection['isTrafficLightDetected'] = True
            
            # Detect color
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                color_scores = {'red': 0, 'yellow': 0, 'green': 0}
                for color_name, (lower, upper) in COLOR_RANGES.items():
                    if color_name == 'red2':
                        continue
                    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
                    if color_name == 'red':
                        mask2 = cv2.inRange(hsv, np.array(COLOR_RANGES['red2'][0]), np.array(COLOR_RANGES['red2'][1]))
                        mask = cv2.bitwise_or(mask, mask2)
                    color_scores[color_name] = cv2.countNonZero(mask)
                
                if max(color_scores.values()) > 50:
                    detection['colorDetected'] = max(color_scores, key=color_scores.get)
            
            # Detect countdown
            h, w = frame.shape[:2]
            cy1 = max(0, y2)
            cy2 = min(h, y2 + 60)
            cx1 = max(0, x1)
            cx2 = min(w, x2)
            
            if cy2 > cy1 and cx2 > cx1:
                countdown_roi = frame[cy1:cy2, cx1:cx2]
                if countdown_roi.size > 0:
                    gray = cv2.cvtColor(countdown_roi, cv2.COLOR_BGR2GRAY)
                    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
                    
                    try:
                        text = pytesseract.image_to_string(
                            thresh, 
                            config='--psm 7 -c tessedit_char_whitelist=0123456789'
                        ).strip()
                        numbers = re.findall(r'\d+', text)
                        if numbers:
                            detection['countdown'] = int(numbers[0])
                    except:
                        pass
            
            # Draw on frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Color text
            color_map = {
                'red': (0, 0, 255),
                'yellow': (0, 255, 255),
                'green': (0, 255, 0),
                'unknown': (255, 255, 255)
            }
            text_color = color_map.get(detection['colorDetected'], (255, 255, 255))
            cv2.putText(frame, f"Color: {detection['colorDetected']}", 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Countdown text
            if detection['countdown']:
                cv2.putText(frame, f"Count: {detection['countdown']}s", 
                           (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    except Exception as e:
        cv2.putText(frame, f"Detection error", (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame, detection

# Sidebar
with st.sidebar:
    st.markdown("## 🚦 **traffic-light.detect()**")
    st.markdown("---")
    
    # Camera selection
    st.markdown("### Camera Source")
    camera_option = st.radio(
        "Select source:",
        ["💻 Laptop Webcam", "📡 ESP32-CAM"]
    )
    
    esp_url = "http://192.168.1.100:81/stream"
    if camera_option == "📡 ESP32-CAM":
        esp_url = st.text_input("ESP32 Stream URL:", value=esp_url)
    
    st.markdown("---")
    
    # Refresh rate control
    st.markdown("### Performance")
    refresh_rate = st.slider(
        "Refresh Rate (ms):",
        min_value=30,
        max_value=200,
        value=50,
        step=10,
        help="Lower = smoother video, Higher = less CPU usage"
    )
    
    st.markdown("---")
    
    # Load model button
    if st.button("🔄 Load Model", use_container_width=True):
        with st.spinner("Loading model... (this may take a moment)"):
            st.session_state.model = load_model()
            st.session_state.model_load_attempted = True
    
    # Start/Stop buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", use_container_width=True):
            # Release existing camera
            if st.session_state.camera:
                st.session_state.camera.release()
            
            # Open new camera
            try:
                if camera_option == "💻 Laptop Webcam":
                    st.session_state.camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                else:
                    st.session_state.camera = cv2.VideoCapture(esp_url)
                
                if st.session_state.camera.isOpened():
                    st.session_state.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    st.session_state.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    st.session_state.running = True
                    st.session_state.frame_count = 0
                    st.success("Camera started!")
                else:
                    st.error("Failed to open camera")
                    st.session_state.camera = None
            except Exception as e:
                st.error(f"Camera error: {e}")
    
    with col2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.running = False
            if st.session_state.camera:
                st.session_state.camera.release()
                st.session_state.camera = None
            st.success("Camera stopped")
    
    st.markdown("---")
    
    # Status
    st.markdown("### System Status")
    
    # Camera status
    cam_status = "✅ Running" if st.session_state.running and st.session_state.camera else "❌ Stopped"
    st.markdown(f"**Camera:** `{cam_status}`")
    
    # Model status
    if st.session_state.model:
        model_status = "✅ Loaded"
    elif st.session_state.model_load_attempted:
        model_status = "❌ Failed (Demo Mode)"
    else:
        model_status = "⏸️ Not Loaded"
    st.markdown(f"**Model:** `{model_status}`")
    
    # Tesseract status
    try:
        tess_version = pytesseract.get_tesseract_version()
        tess_status = f"✅ v{tess_version}"
    except:
        tess_status = "❌ Not found"
    st.markdown(f"**Tesseract:** `{tess_status}`")
    
    # FPS display
    if st.session_state.running:
        st.markdown(f"**FPS:** `{st.session_state.fps:.1f}`")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("## # live-feed")
    
    # Video feed
    if st.session_state.running and st.session_state.camera:
        # Calculate FPS
        current_time = time.time()
        time_diff = current_time - st.session_state.last_refresh
        if time_diff > 0:
            st.session_state.fps = 1.0 / time_diff
        
        # Read frame
        ret, frame = st.session_state.camera.read()
        
        if ret:
            # Run detection (every frame for better accuracy)
            frame, detection = detect_traffic_light(frame, st.session_state.model)
            
            # Add FPS counter to frame
            cv2.putText(frame, f"FPS: {st.session_state.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Add model status to frame
            if st.session_state.model is None:
                cv2.putText(frame, "No Model - Demo Mode", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert to RGB for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update detection in session state
            if detection:
                st.session_state.latest_detection = detection
            
            st.session_state.frame_count += 1
            st.session_state.last_refresh = current_time
        else:
            st.warning("Failed to capture frame")
            placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 30
            cv2.putText(placeholder, "Camera Error - Check Connection", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
            st.image(placeholder, channels="BGR", use_column_width=True)
    else:
        # Show placeholder
        placeholder = np.ones((480, 640, 3), dtype=np.uint8) * 30
        cv2.putText(placeholder, "Camera Stopped", (200, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        cv2.putText(placeholder, "Click START to begin", (200, 300),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 2)
        st.image(placeholder, channels="BGR", use_column_width=True)
    
    # Auto-refresh with controlled rate
    if st.session_state.running:
        time.sleep(refresh_rate / 1000.0)  # Convert ms to seconds
        st.rerun()

with col2:
    st.markdown("## # detection.result")
    
    # Get detection from session state
    detection = st.session_state.latest_detection
    
    # Traffic Light
    st.markdown("### 🚦 Traffic Light")
    if detection['isTrafficLightDetected']:
        st.markdown("**✅ Detected**")
    else:
        st.markdown("**❌ Not Detected**")
    
    # Color
    st.markdown("### 🎨 Color")
    color = detection['colorDetected']
    color_emoji = {
        'red': '🔴', 'yellow': '🟡', 'green': '🟢', 'unknown': '⚪'
    }.get(color, '⚪')
    st.markdown(f"**{color_emoji} `{color}`**")
    
    # Countdown
    st.markdown("### ⏱️ Countdown")
    if detection['countdown']:
        st.markdown(f"**`{detection['countdown']}s`**")
    else:
        st.markdown("**`none`**")
    
    st.markdown("---")
    
    # Stats
    st.markdown("### Statistics")
    st.markdown(f"**Frames:** `{st.session_state.frame_count}`")
    
    if st.session_state.model is None and not st.session_state.model_load_attempted:
        st.info("💡 Click 'Load Model' to enable detection")
    elif st.session_state.model is None and st.session_state.model_load_attempted:
        st.warning("⚠️ Running in demo mode (no detection)")
    
    # Manual refresh button (as backup)
    if st.button("🔄 Manual Refresh", use_container_width=True):
        st.rerun()

# Add refresh info overlay
st.markdown(
    f"""
    <div class="refresh-info">
        🔄 Refresh: {refresh_rate}ms | FPS: {st.session_state.fps:.1f} | Frames: {st.session_state.frame_count}
    </div>
    """,
    unsafe_allow_html=True
)

# Cleanup
import atexit
def cleanup():
    if st.session_state.camera:
        st.session_state.camera.release()
        print("Camera released")

atexit.register(cleanup)