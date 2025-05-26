import cv2
import pytesseract
import os

def process_images_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {filename}: could not read.")
                continue
            
# Load image (e.g., from your camera or dataset)
img = cv2.imread('test2.png')
img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
# Preprocess (convert to grayscale and apply thresholding)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# Extract text
custom_config = r'--oem 3 --psm 6 outputbase digits'  # Digits-only mode
text = pytesseract.image_to_string(thresh, config=custom_config)

print("Detected text:", text)