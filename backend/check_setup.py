import cv2
import torch
from ultralytics import YOLO
import pytesseract
import numpy as np
import sys





print("🔍 Checking your setup...\n")

# Check OpenCV
print("📷 OpenCV version:", cv2.__version__)

# Check PyTorch
print("🔥 PyTorch version:", torch.__version__)
print("   CUDA available:", torch.cuda.is_available())

# Check Tesseract
try:
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    version = pytesseract.get_tesseract_version()
    print(f"🔤 Tesseract version: {version}")
except Exception as e:
    print(f"❌ Tesseract error: {e}")

# Check YOLO
try:
    print("\n🎯 Testing YOLO load...")
    import torch.nn as nn
    from ultralytics.nn.tasks import DetectionModel
    
    safe_classes = [DetectionModel, nn.Module, nn.Sequential, dict, list]
    
    with torch.serialization.safe_globals(safe_classes):
        model = YOLO('yolov8n.pt')
    print("   ✅ YOLO loaded successfully")
except Exception as e:
    print(f"   ❌ YOLO load failed: {e}")

# Check webcam
print("\n📹 Testing webcam...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        print("   ✅ Webcam working")
        print("   Frame shape:", frame.shape)
    else:
        print("   ❌ Could not read from webcam")
    cap.release()
else:
    print("   ❌ Could not open webcam")

print("\n✅ Setup check complete!")