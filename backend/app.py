from flask import Flask, Response, jsonify, request
from flask_cors import CORS
import cv2
import threading
import time
import numpy as np

from camera import CameraManager
from detector import TrafficLightDetector

app = Flask(__name__)
# Enable CORS for all routes
CORS(app, origins=["http://localhost:3000"])  # Allow only Next.js frontend

# Global instances
camera = CameraManager(use_esp=False)          # start with webcam
detector = TrafficLightDetector()
latest_detection = {}                           # stores the latest JSON result
lock = threading.Lock()

def generate_frames():
    global latest_detection
    frame_count = 0
    while True:
        try:
            frame = camera.get_frame()
            if frame is None:
                # If no frame, create a blank frame with error message
                frame = 255 * np.ones((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No Camera Signal", (200, 240),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Run detection every few frames to save CPU
                if frame_count % 3 == 0:  # Detect every 3rd frame
                    detection_result = detector.detect(frame)
                    
                    # Update latest detection (thread‑safe)
                    with lock:
                        latest_detection = detection_result
                
                frame_count += 1

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            time.sleep(0.03)  # ~30 fps
            
        except Exception as e:
            print(f"Error in generate_frames: {e}")
            time.sleep(0.1)

@app.route('/video_feed')
def video_feed():
    """MJPEG stream route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame',
                    headers={'Cache-Control': 'no-cache, no-store, must-revalidate',
                            'Pragma': 'no-cache',
                            'Expires': '0'})

@app.route('/api/detection', methods=['GET'])
def get_detection():
    """REST API returning latest detection JSON"""
    with lock:
        # Ensure we always return at least the base structure
        response = {
            'isTrafficLightDetected': latest_detection.get('isTrafficLightDetected', False),
            'colorDetected': latest_detection.get('colorDetected', 'unknown'),
            'isCountDownVisible': latest_detection.get('isCountDownVisible', False),
            'countdown': latest_detection.get('countdown', None)
        }
        return jsonify(response)

@app.route('/api/switch_camera', methods=['POST', 'OPTIONS'])
def switch_camera():
    """Switch between webcam and ESP32-CAM"""
    # Handle preflight OPTIONS request for CORS
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
            
        target = data.get('target')  # 'webcam' or 'esp'
        
        if target == 'webcam':
            # Run switch in a separate thread to avoid blocking
            def switch():
                camera.switch_to_webcam()
            threading.Thread(target=switch).start()
            return jsonify({'status': 'switched to webcam'})
            
        elif target == 'esp':
            def switch():
                camera.switch_to_esp()
            threading.Thread(target=switch).start()
            return jsonify({'status': 'switched to ESP32-CAM'})
            
        else:
            return jsonify({'error': 'invalid target'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    camera_status = 'ok' if camera.cap and camera.cap.isOpened() else 'error'
    return jsonify({
        'status': 'ok',
        'camera': camera_status,
        'source': 'esp' if camera.use_esp else 'webcam'
    })

if __name__ == '__main__':
    print("Starting Traffic Light Detection Server...")
    print("Access the video feed at: http://localhost:5000/video_feed")
    print("Access the detection API at: http://localhost:5000/api/detection")
    print("Press Ctrl+C to stop")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)