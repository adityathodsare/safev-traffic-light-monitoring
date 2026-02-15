import cv2
import numpy as np
import pytesseract
from PIL import Image
from ultralytics import YOLO
import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
import re
import os

# Tesseract path configuration
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Verify Tesseract is working
try:
    tesseract_version = pytesseract.get_tesseract_version()
    print(f"✅ Tesseract OCR initialized: {tesseract_version}")
except Exception as e:
    print(f"⚠️ Tesseract OCR warning: {e}")
    print("   Countdown detection may not work properly")

class TrafficLightDetector:
    def __init__(self, model_path='yolov8n.pt'):
        # Comprehensive safe globals for YOLO model loading
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
            nn.MaxPool2d,
            nn.AdaptiveAvgPool2d,
            nn.LeakyReLU,
            dict,
            list,
            tuple,
            set,
            str,
            int,
            float,
            bool,
            type(None),
        ]
        
        # Add common container types
        try:
            from torch.nn.modules.container import Sequential, ModuleList
            safe_classes.extend([Sequential, ModuleList])
        except ImportError:
            pass
            
        # Load model with safe globals context
        try:
            with torch.serialization.safe_globals(safe_classes):
                self.model = YOLO(model_path)
            print("✅ YOLO model loaded successfully")
        except Exception as e:
            print(f"⚠️ YOLO model loading error: {e}")
            print("   Attempting to download fresh model...")
            # Try downloading fresh model
            self.model = YOLO('yolov8n.pt')
        
        # COCO class 9 is "traffic light"
        self.target_class = 9

        # Colour ranges in HSV
        self.color_ranges = {
            'red': [(0, 100, 100), (10, 255, 255)],      # lower red
            'red2': [(160, 100, 100), (180, 255, 255)],  # upper red (red wraps around)
            'yellow': [(20, 100, 100), (30, 255, 255)],
            'green': [(40, 100, 100), (80, 255, 255)]
        }

    def detect(self, frame):
        """
        Returns a dictionary with detection results.
        """
        # Run YOLO inference
        results = self.model(frame, classes=[self.target_class], conf=0.3)  # confidence threshold

        # Default response
        output = {
            'isTrafficLightDetected': False,
            'colorDetected': 'unknown',
            'isCountDownVisible': False,
            'countdown': None
        }

        if len(results[0].boxes) == 0:
            return output

        # Take the largest traffic light (by area)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        largest_idx = np.argmax(areas)
        x1, y1, x2, y2 = map(int, boxes[largest_idx])

        # Crop the traffic light region
        tl_roi = frame[y1:y2, x1:x2]
        if tl_roi.size == 0:
            return output

        output['isTrafficLightDetected'] = True

        # 1. Detect colour inside the bounding box
        color = self._detect_color(tl_roi)
        output['colorDetected'] = color

        # 2. Detect countdown just below the traffic light
        countdown_value, countdown_visible = self._detect_countdown(frame, (x1, y2, x2, y2 + 80))
        output['isCountDownVisible'] = countdown_visible
        output['countdown'] = countdown_value

        # Draw annotations on the frame
        self._draw_annotations(frame, (x1, y1, x2, y2), color, countdown_value)

        return output

    def _detect_color(self, roi):
        """
        roi: cropped image of the traffic light.
        Returns 'red', 'yellow', 'green', or 'unknown'.
        """
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        height = roi.shape[0]

        # Divide the traffic light into three vertical sections
        section_height = max(1, height // 3)
        sections = {
            'top': hsv[0:section_height, :],
            'middle': hsv[section_height:2*section_height, :] if height >= 2*section_height else hsv[section_height:height, :],
            'bottom': hsv[2*section_height:3*section_height, :] if height >= 3*section_height else hsv[2*section_height:height, :]
        }

        # For each section, count pixels that match each colour
        color_counts = {'red': 0, 'yellow': 0, 'green': 0}
        for sect_name, sect_hsv in sections.items():
            if sect_hsv.size == 0:
                continue
            for color_name, (lower, upper) in self.color_ranges.items():
                if color_name == 'red2':  # Skip red2 in loop, handle separately
                    continue
                mask = cv2.inRange(sect_hsv, np.array(lower), np.array(upper))
                # Also check the second red range for red color
                if color_name == 'red':
                    mask2 = cv2.inRange(sect_hsv, np.array(self.color_ranges['red2'][0]), np.array(self.color_ranges['red2'][1]))
                    mask = cv2.bitwise_or(mask, mask2)
                count = cv2.countNonZero(mask)
                color_counts[color_name] += count

        # Find the colour with the highest count
        total_pixels = sum(color_counts.values())
        if total_pixels < 50:  # noise threshold
            return 'unknown'
            
        detected = max(color_counts, key=color_counts.get)
        return detected

    def _detect_countdown(self, frame, countdown_region):
        """
        countdown_region: (x1, y1, x2, y2) just below the traffic light.
        Returns (number, is_visible)
        """
        x1, y1, x2, y2 = countdown_region
        # Ensure coordinates are within frame
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

        # Preprocess the image for better OCR
        # Convert to grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # Optional: Apply morphological operations to enhance digits
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # Use pytesseract to extract digits with optimized configuration
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
        
        try:
            text = pytesseract.image_to_string(thresh, config=custom_config).strip()
            
            # Try to parse an integer
            if text.isdigit():
                return int(text), True
                
            # If there are multiple digits, take the first group
            numbers = re.findall(r'\d+', text)
            if numbers:
                return int(numbers[0]), True
                
            # Also try with original grayscale if thresholding didn't work
            text2 = pytesseract.image_to_string(gray, config=custom_config).strip()
            if text2.isdigit():
                return int(text2), True
            numbers2 = re.findall(r'\d+', text2)
            if numbers2:
                return int(numbers2[0]), True
                
        except Exception as e:
            print(f"OCR error: {e}")
            
        return None, False

    def _draw_annotations(self, frame, bbox, color, countdown):
        x1, y1, x2, y2 = bbox
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw color text
        color_text = f"Color: {color}"
        color_map = {
            'red': (0, 0, 255),
            'yellow': (0, 255, 255),
            'green': (0, 255, 0),
            'unknown': (255, 255, 255)
        }
        text_color = color_map.get(color, (255, 255, 255))
        cv2.putText(frame, color_text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Draw countdown if detected
        if countdown is not None:
            countdown_text = f"Count: {countdown}s"
            cv2.putText(frame, countdown_text, (x1, y2+25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)