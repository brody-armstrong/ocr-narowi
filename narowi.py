import os
import re

import cv2
import pandas as pd
import pytesseract

def process_images_from_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Skipping {filename}: could not read.")
                continue
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: could not read image at {image_path}")
        return None

    # 3.a. Image resizing (2x upscaling, linear interpolation)
    img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # 3.b. Noise reduction (Gaussian blur)
    img_blurred = cv2.GaussianBlur(img_resized, (5, 5), 0)

    # 3.c. Convert to grayscale
    gray_img = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)

    # 3.d. Adaptive histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(gray_img)

    # 3.f. Adaptive thresholding for binarization
    # Block size and C value might need tuning
    binary_img = cv2.adaptiveThreshold(img_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)

    # 3.g. Morphological operations (opening then closing)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    img_opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    img_closed = cv2.morphologyEx(img_opened, cv2.MORPH_CLOSE, kernel)
    
    return img_closed

def extract_numeric_value(text_match):
    """
    Tries to convert a text match to a float or int, stripping common units.
    Returns the numeric value, or None if conversion fails.
    """
    # Remove common units and trailing non-numeric characters
    cleaned_text = re.sub(r'[^\d\.-]$', '', text_match.replace('%', '').replace('°C', '').replace('°F', ''))
    try:
        return float(cleaned_text)
    except ValueError:
        try:
            return int(cleaned_text)
        except ValueError:
            return None

def extract_numbers_with_confidence(ocr_results):
    """
    Extracts numbers, percentages, and temperatures from OCR results.
    ocr_results: A list of dictionaries, each with 'text', 'confidence', 'bbox'.
    Returns a list of dictionaries with extracted numeric data.
    """
    numbers = []
    patterns = [
        r'-?\d+\.\d+',  # Decimals
        r'-?\d+',       # Integers
        r'\d+\.\d*%',  # Percentages
        r'-?\d+\.?\d*[°][CF]'  # Temperatures
    ]

    for item in ocr_results:
        text = item['text']
        for pattern in patterns:
            for match in re.findall(pattern, text):
                numeric_value = extract_numeric_value(match)
                if numeric_value is not None:
                    numbers.append({
                        'value': numeric_value,
                        'raw_text': match,
                        'confidence': item['confidence'], # Corrected key from 'confidence' to 'conf' based on pytesseract
                        'bbox': item['bbox']
                    })
    return numbers

def perform_ocr(processed_image):
    """
    Performs OCR on a preprocessed image using Tesseract with multiple configurations.
    Returns a list of dictionaries, each containing 'text', 'confidence', and 'bbox'.
    """
    if processed_image is None:
        print("Error: No image provided to perform_ocr.")
        return []

    number_config = '--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.-%°CF'
    fallback_configs = [
        '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.-%°CF',
        '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.-',
        '--oem 3 --psm 10'
    ]

    ocr_data_df = None
    configs_to_try = [number_config] + fallback_configs

    for i, config in enumerate(configs_to_try):
        try:
            print(f"Attempting OCR with config {i+1}: {config}")
            ocr_data_df = pytesseract.image_to_data(processed_image, config=config, output_type=pytesseract.Output.DATAFRAME)
            
            # Check for usable results: non-empty dataframe and some confidence > 0
            if ocr_data_df is not None and not ocr_data_df.empty:
                # Filter out rows where 'conf' is not a valid number (e.g. header row from tesseract output)
                ocr_data_df_filtered = ocr_data_df[pd.to_numeric(ocr_data_df['conf'], errors='coerce').notnull()]
                if not ocr_data_df_filtered.empty and ocr_data_df_filtered['conf'].astype(float).max() > 0:
                    print(f"OCR successful with config: {config}")
                    break  # Successful OCR, exit loop
                else:
                    print(f"Config {config} yielded no usable results (all conf <= 0 or empty text).")
                    ocr_data_df = None # Reset for next try
            else:
                print(f"Config {config} yielded empty dataframe.")
                ocr_data_df = None # Reset for next try
        except Exception as e:
            print(f"Error during OCR with config {config}: {e}")
            ocr_data_df = None # Reset for next try
            continue # Try next config

    if ocr_data_df is None or ocr_data_df.empty:
        print("All OCR configurations failed to produce usable results.")
        return []

    results = []
    # Filter out rows with low confidence or empty text strings
    # Also ensure 'text' is a string and 'conf' is numeric before filtering.
    ocr_data_df = ocr_data_df[pd.to_numeric(ocr_data_df['conf'], errors='coerce').notnull()] # Ensure conf is numeric
    ocr_data_df['conf'] = ocr_data_df['conf'].astype(float)
    ocr_data_df['text'] = ocr_data_df['text'].astype(str)

    # Filter based on confidence and non-empty, non-whitespace text
    # Using word_num > 0 to filter out block-level results and only keep word-level results.
    # Using line_num > 0 might also be useful. block_num > 0 as well.
    # Tesseract's image_to_data includes entries for page, block, paragraph, line, and word.
    # We are typically interested in word-level data. Word level data has word_num > 0.
    # Block level data has block_num > 0, par_num == 0, line_num == 0, word_num == 0.
    # We also filter out text that is only whitespace.
    filtered_df = ocr_data_df[(ocr_data_df['conf'] >= 10) & (ocr_data_df['text'].str.strip() != '') & (ocr_data_df['word_num'] > 0)]


    for index, row in filtered_df.iterrows():
        results.append({
            'text': str(row['text']).strip(),
            'confidence': float(row['conf']),
            'bbox': [int(row['left']), int(row['top']), int(row['width']), int(row['height'])]
        })
    
    if not results:
        print("No text segments met the confidence threshold or other filtering criteria.")

    return results

def identify_device_type(image):
    """
    Placeholder function for device type identification.
    Takes an image object (currently unused) and returns a placeholder string.
    """
    # In the future, this function will use image analysis or other methods
    # to determine the type of device shown in the image.
    # For now, it's a placeholder.
    # The 'image' argument is kept for future compatibility.
    _ = image # Mark as unused for linters
    return "Device type identification not yet implemented."

# The following global image processing code is now commented out
# as its functionality will be incorporated into the preprocess_image function
# and the main script logic.

# # Load image (e.g., from your camera or dataset)
# img = cv2.imread('test2.png')
# img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
# # Preprocess (convert to grayscale and apply thresholding)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
# 
# # Extract text
# custom_config = r'--oem 3 --psm 6 outputbase digits'  # Digits-only mode
# text = pytesseract.image_to_string(thresh, config=custom_config)
# 
# print("Detected text:", text)

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python narowi.py <folder_path>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"Error: Folder not found at {folder_path}")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        print(f"Error: Path provided is not a folder — {folder_path}")
        sys.exit(1)

    print(f"Processing folder: {folder_path}")
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(image_extensions)]

    if not image_files:
        print("No image files found in the folder.")
        sys.exit(0)

    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        print(f"\n---\nProcessing image: {image_path}")

        preprocessed_image = preprocess_image(image_path)
        if preprocessed_image is None:
            print("Skipping this image due to preprocessing error.")
            continue

        ocr_data = perform_ocr(preprocessed_image)
        if not ocr_data:
            print("No OCR text found.")
            continue

        extracted_numbers = extract_numbers_with_confidence(ocr_data)
        if extracted_numbers:
            print("\nExtracted Numbers:")
            for num_info in extracted_numbers:
                print(
                    f"  Value: {num_info['value']}, "
                    f"Raw Text: '{num_info['raw_text']}', "
                    f"Confidence: {num_info['confidence']:.2f}%, "
                    f"BBox: {num_info['bbox']}"
                )
        else:
            print("No numbers extracted or matched the criteria.")

        device_type_info = identify_device_type(preprocessed_image)
        print(f"Device Type Info: {device_type_info}")

    print("\nAll images processed.")
