import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

print("🔍 Testing Countdown Detection...")

# Create a test image with a number (simulating countdown)
test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255  # White background

# Add text to the image
from PIL import Image, ImageDraw, ImageFont
pil_image = Image.fromarray(test_image)
draw = ImageDraw.Draw(pil_image)

# Try to use a font, or fall back to default
try:
    font = ImageFont.truetype("arial.ttf", 48)
except:
    font = ImageFont.load_default()

draw.text((50, 20), "42", fill=(0, 0, 0), font=font)
test_image = np.array(pil_image)

# Convert to grayscale
gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

# Apply thresholding
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Try OCR
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789'
text = pytesseract.image_to_string(thresh, config=custom_config).strip()

print(f"OCR Result: '{text}'")
if text == "42":
    print("✅ Countdown detection working correctly!")
else:
    print("⚠️ OCR may need calibration")

# Show the test image (optional)
cv2.imshow("Test Image", test_image)
cv2.imshow("Thresholded", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()