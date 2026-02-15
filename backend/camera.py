import cv2
import time

class CameraManager:
    def __init__(self, use_esp=False, esp_url=None):
        self.use_esp = use_esp
        # Default ESP32-CAM stream URL - CHANGE THIS TO YOUR ESP32-CAM IP
        self.esp_url = esp_url or "http://192.168.1.100:81/stream"
        self.cap = None
        self._open_camera()

    def _open_camera(self):
        # Release existing camera if any
        if self.cap is not None:
            self.cap.release()
            time.sleep(0.5)  # Give it time to release
        
        try:
            if self.use_esp:
                print(f"Connecting to ESP32-CAM at {self.esp_url}...")
                # ESP32-CAM stream (usually MJPEG over HTTP)
                self.cap = cv2.VideoCapture(self.esp_url)
            else:
                print("Opening laptop webcam...")
                # Laptop webcam (usually 0)
                self.cap = cv2.VideoCapture(0)
            
            # Give camera time to warm up
            time.sleep(2)
            
            # Check if camera opened successfully
            if not self.cap.isOpened():
                print("Failed to open camera!")
                self.cap = None
            else:
                print("Camera opened successfully")
        except Exception as e:
            print(f"Error opening camera: {e}")
            self.cap = None

    def switch_to_webcam(self):
        print("Switching to webcam...")
        self.use_esp = False
        self._open_camera()

    def switch_to_esp(self):
        print("Switching to ESP32-CAM...")
        self.use_esp = True
        self._open_camera()

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to grab frame")
            return None
        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            print("Camera released")