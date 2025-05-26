import cv2
import pytesseract
import os
import numpy as np
import argparse
import logging
from collections import Counter
import re
import json
    "blood_pressure_monitor": ["sys", "dia", "pul", "mmhg"],
    "oximeter": ["spo2", "pr", "pi", "%"],
    "thermometer": ["°c", "°f", "temp", "body", "therm"],
    "weight_scale": ["kg", "lb", "weight", "scale"]
}

def identify_device_type(ocr_text: str) -> str:
    ocr_text_lower = ocr_text.lower()
    matches = Counter()

    for device_type, keywords in DEVICE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in ocr_text_lower:
                matches[device_type] += 1
    
    if not matches:
        return "unknown_device"
    
    # Return the device type with the most matches
    # If there's a tie, it returns one of the top ones based on internal ordering
    return matches.most_common(1)[0][0]

READING_LABELS = {
    "blood_pressure_monitor": {
        "sys": "systolic", "dia": "diastolic", "pul": "pulse", "map": "mean_arterial_pressure"
    },
    "oximeter": {
        "spo2": "spo2", "pr": "pulse_rate", "pi": "perfusion_index"
    },
    "thermometer": {
        "temp": "temperature", "body": "temperature" # Assuming 'body' implies temperature
    },
    "weight_scale": {
        "kg": "weight_kg", "lb": "weight_lb", "weight": "weight"
    }
}

PHYSIOLOGICAL_RANGES = {
    "systolic": {"min": 70, "max": 190, "unit": "mmHg"},
    "diastolic": {"min": 40, "max": 100, "unit": "mmHg"},
    "pulse": {"min": 40, "max": 180, "unit": "bpm"},
    "pulse_rate": {"min": 40, "max": 180, "unit": "bpm"}, # Oximeter often calls it pulse_rate
    "spo2": {"min": 85, "max": 100, "unit": "%"},
    "temperature": {"min": 35.0, "max": 42.0, "unit": "°C"}, # Assuming Celsius for now
    "weight_kg": {"min": 1, "max": 250, "unit": "kg"},    # Example for weight
    "weight_lb": {"min": 2, "max": 550, "unit": "lb"},   # Example for weight
    # perfusion_index (pi) has a wide range and might not be suitable for strict min/max like this
    # "perfusion_index": {"min": 0.02, "max": 20, "unit": "%"},
}

def check_physiological_range(metric_name: str, value: float, errors_list: list):
    """
    Checks if a given value for a metric is within a plausible physiological range.
    Appends a warning to errors_list if it's outside the range.
    """
    if metric_name in PHYSIOLOGICAL_RANGES:
        constraints = PHYSIOLOGICAL_RANGES[metric_name]
        min_val = constraints["min"]
        max_val = constraints["max"]
        unit = constraints["unit"]
        if not (min_val <= value <= max_val):
            warning_msg = (
                f"Warning: {metric_name} value {value} {unit} is outside the typical range of {min_val}-{max_val} {unit}."
            )
            errors_list.append(warning_msg)
            logging.warning(warning_msg) # Also log it as a warning

def classify_readings(sorted_detailed_rois: list, device_type: str) -> dict: # Parameter name updated
    result = {
        "readings": {}, 
        "confidence_scores": {}, 
        "errors": []
    }
    potential_numbers = []
    potential_labels = []

    for item in sorted_detailed_rois: # Iterating through new structure
        text_roi = item['text']
        bbox = item['bbox']
        confidence = item['confidence']
        text_roi_lower = text_roi.lower()

        # Attempt to extract a number
        # Regex to find integers or decimals
        match = re.search(r"(\d+\.?\d*)", text_roi)
        if match:
            number_str = match.group(1)
            try:
                # Attempt to convert to int if no decimal, else float
                number = int(number_str) if '.' not in number_str else float(number_str)
                potential_numbers.append({
                    'text': text_roi, # Original text of the ROI item
                    'value': number, 
                    'bbox': bbox, 
                    'confidence': confidence # Storing confidence
                })
                logging.debug(f"Extracted number: {number} (Conf: {confidence}) from ROI: '{text_roi}' at {bbox}")
            except ValueError:
                error_msg = f"Could not convert extracted number '{number_str}' to float/int from ROI: '{text_roi}' (Conf: {confidence}) at {bbox}"
                logging.warning(error_msg)
                result["errors"].append(error_msg)

        # Attempt to identify a label
        if device_type != "unknown_device" and device_type in READING_LABELS:
            keywords_map = READING_LABELS[device_type]
            for label_keyword, metric_name in keywords_map.items():
                if label_keyword in text_roi_lower:
                    potential_labels.append({
                        'text': text_roi, # Original text of the ROI item
                        'bbox': bbox, 
                        'label_keyword': label_keyword, 
                        'metric_name': metric_name,
                        'confidence': confidence # Confidence of the label text itself
                    })
                    logging.debug(f"Identified label: '{label_keyword}' (metric: '{metric_name}', Conf: {confidence}) in ROI: '{text_roi}' at {bbox}")

    # TODO: Implement logic to associate numbers with labels based on proximity or layout.
    # This logic will populate result["readings"] and result["confidence_scores"].
    # For now, potential_numbers and potential_labels are collected but not fully processed into the result dict.
    # Example of how it might be used (actual implementation is for a future subtask):
    # if potential_numbers and potential_labels: # Simplified example
    #   for num_entry in potential_numbers: # Iterate through all found numbers
    #       for label_entry in potential_labels: # Iterate through all found labels
    #           # TODO: Implement actual proximity/layout based association logic here
    #           # This is a conceptual placeholder showing where association would happen:
    #           is_associated = False # Replace with real association check
    #           if is_associated: 
    #               metric_name = label_entry['metric_name']
    #               value = num_entry['value']
    #               confidence = num_entry['confidence']
    #
    #               result["readings"][metric_name] = value
    #               result["confidence_scores"][metric_name] = confidence
    #
    #               # Perform range check
    #               check_physiological_range(metric_name, value, result["errors"])
    #               break # Assuming one number associates with one label for simplicity here
    #       # if num_entry was associated and we broke, continue to next num_entry or handle as needed

    # For debugging, we can temporarily add collected items to the result:
    # result['potential_numbers_debug'] = potential_numbers
    # result['potential_labels_debug'] = potential_labels
    
    return result

def process_images_from_folder(folder_path):

    custom_config = r'--oem 3 --psm 11'
    all_image_data = []

    if not os.path.exists(folder_path):
        logging.error(f"Folder not found at {folder_path}")
        return all_image_data # Return empty list if folder not found
        
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            logging.info(f"Processing {filename}...")
            img = cv2.imread(img_path)
            if img is None:
                logging.error(f"Couldnt not read of decode image: {filename}")
                continue
            

            try:
                resized_img = cv2.resize(img, None, fx = 2, fy = 2, interpolation = cv2.INTER_LINEAR)

                gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)

                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                clahe_output = clahe.apply(gray)

                # Apply sharpening
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                sharpened_img = cv2.filter2D(clahe_output, -1, kernel)

                blurred = cv2.GaussianBlur(sharpened_img, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                               cv2.THRESH_BINARY_INV, 11, 2)
                countours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detailed_rois = [] # Changed from rois_with_text

                for i, countour in enumerate(countours):
                    x, y, w, h = cv2.boundingRect(countour) # x, y are coordinates of the ROI's top-left corner in the thresholded image
                    if w > 10 and h > 10 and w < thresh.shape[1]: # Basic filtering for ROI size
                        roi = thresh[y:y+h, x:x+w] # This is the sub-image (the ROI)

                        try:
                            # Using image_to_data to get detailed OCR output including confidence
                            ocr_data = pytesseract.image_to_data(roi, config=custom_config, output_type=pytesseract.Output.DICT)
                            
                            n_boxes = len(ocr_data['level'])
                            for j in range(n_boxes):
                                word_text = ocr_data['text'][j].strip()
                                confidence = int(ocr_data['conf'][j])
                                
                                # Filter for meaningful words and reasonable confidence
                                # Using 40 as a threshold for confidence as suggested, can be tuned
                                if word_text and confidence > 0: # Changed from 40 to 0 as per instruction "confidence > 0"
                                    # Bounding box from Tesseract is relative to the ROI (sub-image)
                                    word_left = ocr_data['left'][j]
                                    word_top = ocr_data['top'][j]
                                    word_width = ocr_data['width'][j]
                                    word_height = ocr_data['height'][j]
                                    
                                    # Calculate absolute bounding box relative to the original image (or rather, the resized_img)
                                    # (x, y) is the top-left of the ROI itself.
                                    absolute_bbox = (x + word_left, y + word_top, word_width, word_height)
                                    
                                    detailed_rois.append({
                                        'text': word_text,
                                        'confidence': confidence,
                                        'bbox': absolute_bbox # Storing the absolute bbox
                                    })
                                    logging.debug(f"ROI #{i+1}, Word #{j+1}: '{word_text}' (Conf: {confidence}) at {absolute_bbox}")

                        except pytesseract.TesseractError as e:
                            logging.error(f"Tesseract OCR (image_to_data) failed for an ROI in {filename} (ROI #{i+1}): {e}")
                        except Exception as e: # Catch other potential errors during OCR
                            logging.error(f"An unexpected error occurred during detailed OCR for an ROI in {filename} (ROI #{i+1}): {e}")
                
                # Sort detailed_rois by y then x coordinate of their bounding boxes
                sorted_detailed_rois = sorted(detailed_rois, key=lambda item: (item['bbox'][1], item['bbox'][0]))

                # Reconstruct the text for device identification
                text = " ".join([item['text'] for item in sorted_detailed_rois])
                logging.info(f"Detected text (reconstructed from detailed ROIs) in {filename}: {text}")

                device_type = identify_device_type(text)
                logging.info(f"Identified device type in {filename}: {device_type}")

                # Call classify_readings with sorted_detailed_rois
                # Note: The structure of sorted_detailed_rois is different from the old sorted_rois
                # classify_readings will need to be adapted in a subsequent subtask
                classified_readings_data = classify_readings(sorted_detailed_rois, device_type) 
                
                image_data = {
                    "image_filename": filename,
                    "device_type": device_type,
                    "readings": classified_readings_data.get("readings", {}),
                    "errors": classified_readings_data.get("errors", []),
                    "confidence_scores": classified_readings_data.get("confidence_scores", {})
                }
                logging.info(f"Structured data for {filename}: {image_data}")
                all_image_data.append(image_data)

                if not text:
                    logging.warning(f"No text detected in {filename}.")
                else:
                    # Print statement for overall OCR result (optional, can be removed if too verbose)
                    print(f"\n🧾 OCR result (reconstructed) for {filename}:\n{text}\n")
                
                # Print statement for detailed ROI text (optional, can be removed if too verbose)
                # This replaces the old loop that printed (bbox, roi_text)
                for item in sorted_detailed_rois:
                    print(f"  Detailed ROI: BBox{item['bbox']} → '{item['text']}' (Conf: {item['confidence']})")

            except Exception as e:
                logging.error(f"Failed to process image {filename}: {e}")       
        else:
            if os.path.isfile(img_path): 
                 logging.warning(f"Skipping {filename}: not a recognized image file type (png, jpg, jpeg).")
    return all_image_data

def summarize_image_data(image_data: dict) -> str:
    summary_lines = []
    summary_lines.append(f"--- Summary for {image_data['image_filename']} ---")
    summary_lines.append(f"Device Type: {image_data['device_type']}")

    # Format Readings
    if image_data.get('readings'): # Check if 'readings' key exists and is not empty
        summary_lines.append("Readings:")
        for metric, value in image_data['readings'].items():
            unit = ""
            if metric in PHYSIOLOGICAL_RANGES and 'unit' in PHYSIOLOGICAL_RANGES[metric]:
                unit = PHYSIOLOGICAL_RANGES[metric]['unit']
            
            confidence_val = image_data.get('confidence_scores', {}).get(metric, 'N/A')
            confidence_str = ""
            if isinstance(confidence_val, (int, float)):
                confidence_str = f"{confidence_val:.0f}%" # Format as integer percentage
            else:
                confidence_str = "N/A"
            
            metric_display_name = metric.replace('_', ' ').title()
            summary_lines.append(f"  - {metric_display_name}: {value} {unit} (Confidence: {confidence_str})")
    else:
        summary_lines.append("Readings: No readings detected.")

    # Format Errors/Warnings
    if image_data.get('errors'): # Check if 'errors' key exists and is not empty
        summary_lines.append("Alerts/Errors:")
        for error_msg in image_data['errors']:
            summary_lines.append(f"  - {error_msg}")
    
    return "\n".join(summary_lines) + "\n"

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Process images in a folder to extract text using OCR.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")
    
    try:
        args = parser.parse_args()
        processed_data = process_images_from_folder(args.folder_path)

        if processed_data: # Will be an empty list if folder not found or no images processed
            logging.info("--- Final Structured Output ---")
            for item in processed_data:
                summary = summarize_image_data(item)
                print(summary)
        else:
            logging.info("No data processed or folder not found.")
            
    except SystemExit:
        # Argparse can cause SystemExit (e.g., for --help). This is normal.
        logging.info("Application exited via argparse (e.g., help displayed or argument error).")
    except Exception as e:
        logging.critical(f"An unexpected critical error occurred in the main application flow: {e}")

if __name__ == "__main__":
    main()
