import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import cv2

# Load image with error checking
image_path = 'patient_file.jpg'
image = cv2.imread(image_path)
if image is None:
    raise FileNotFoundError(f"Image not found at: {image_path}")

# Rest of your code (grayscale, thresholding, OCR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
dilated = cv2.dilate(thresh, kernel, iterations=1)
text = pytesseract.image_to_string(dilated, config=r'--oem 3 --psm 6')
print(text)