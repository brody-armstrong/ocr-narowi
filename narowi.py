import cv2
import pytesseract
import os
import re
import logging

# Define Sample Regex Patterns
# These patterns aim to capture numbers (integers or decimals) followed by common units.
# They use named capture groups 'value' and 'unit'.
# Case-insensitive matching for units will be handled by re.IGNORECASE during compilation.
measurement_patterns = [
    r'(?P<value>\d+\.\d+|\d+)\s*(?P<unit>cm|mm|m|km|inch|ft|yd|mi|mg|g|kg|oz|lb|ml|l|C|F|deg C|deg F)\b',  # Number, optional space, unit
    r'(?P<value>\d+\.\d+|\d+)(?P<unit>cm|mm|m|km|inch|ft|yd|mi|mg|g|kg|oz|lb|ml|l|C|F|deg C|deg F)\b'     # Number directly attached to unit
]

def extract_measurements(ocr_text, patterns):
    """
    Extracts measurements from OCR text using a list of regex patterns.

    Args:
        ocr_text (str): The text output from Tesseract.
        patterns (list of str): A list of regular expression patterns.

    Returns:
        list: A list of dictionaries, where each dictionary represents a
              found measurement (e.g., {'value': 12.3, 'unit': 'cm', 'original_text': '12.3 cm'}).
    """
    found_measurements = []
    for pattern_str in patterns:
        # Compile the regex pattern with ignorecase for units
        # Note: The re.IGNORECASE flag applies to the whole pattern.
        # If case sensitivity is needed for values, more complex patterns might be required
        # or separate handling for the 'unit' part. For units, IGNORECASE is generally good.
        try:
            compiled_pattern = re.compile(pattern_str, re.IGNORECASE)
        except re.error as e:
            logging.error(f"Error compiling regex pattern '{pattern_str}': {e}")
            continue

        for match in compiled_pattern.finditer(ocr_text):
            try:
                value_str = match.group('value')
                unit_str = match.group('unit')
                original_text = match.group(0)

                # Attempt to convert value to float
                try:
                    value_float = float(value_str)
                except ValueError:
                    logging.warning(f"Could not convert value '{value_str}' to float for match '{original_text}'. Skipping.")
                    continue

                measurement_data = {
                    'value': value_float,
                    'unit': unit_str.strip(), # Basic stripping, more advanced normalization could be added
                    'original_text': original_text
                }
                
                # Avoid adding duplicate entries if patterns overlap significantly
                # This is a simple check; more sophisticated de-duplication might be needed
                is_duplicate = False
                for fm in found_measurements:
                    if fm['original_text'] == original_text and fm['value'] == value_float and fm['unit'].lower() == unit_str.strip().lower():
                        is_duplicate = True
                        break
                if not is_duplicate:
                    found_measurements.append(measurement_data)
                    
            except IndexError:
                # This happens if 'value' or 'unit' named groups are not in the match
                # Should not happen if patterns are correctly defined with these named groups
                logging.error(f"Pattern '{pattern_str}' matched but did not contain 'value' or 'unit' groups for match '{match.group(0)}'.")
            except Exception as e:
                logging.error(f"An unexpected error occurred while processing match '{match.group(0)}' with pattern '{pattern_str}': {e}")
                
    return found_measurements

def process_images_from_folder(folder_path):
    """
    Processes all images in a given folder, applying OCR and measurement extraction.
    """
    global measurement_patterns # Ensure access to global patterns

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            logging.info(f"Attempting to process image: {filename}")
            
            try:
                img = cv2.imread(img_path)
                if img is None:
                    logging.warning(f"Could not read or decode image: {filename}. Skipping.")
                    continue
                
                # Apply image preprocessing
                img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
                gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

                # Perform OCR using Tesseract
                custom_config = r'--oem 3 --psm 3'
                text = pytesseract.image_to_string(thresh, config=custom_config)
                
                logging.info(f"Detected text for {filename}: {text.strip()}")
                
                # Call the extract_measurements function
                extracted_data = extract_measurements(text, measurement_patterns)
                if not extracted_data and text.strip(): # Log if OCR found text but no measurements
                    logging.info(f"No measurements found in {filename} after OCR, though text was detected.")
                elif not text.strip(): # Log if OCR found no text at all
                    logging.info(f"No text detected by OCR in {filename}.")

                logging.info(f"Extracted measurements for {filename}: {extracted_data}")

            except cv2.error as e:
                logging.error(f"OpenCV error processing {filename}: {e}")
            except pytesseract.TesseractError as e:
                logging.error(f"Tesseract error processing {filename}: {e}")
            except Exception as e:
                logging.error(f"Generic error processing {filename}: {e}")
            finally:
                logging.info(f"Finished processing attempt for {filename}.")
                print("-" * 40) # Keep some console separation for readability

# Main execution block
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Comment out or remove the existing single-image processing logic
    # img = cv2.imread('test2.png')
    # img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2) # Adaptive thresholding
    # custom_config = r'--oem 3 --psm 3'  # Allow all characters
    # text = pytesseract.image_to_string(thresh, config=custom_config)
    # print("Detected text:", text.strip()) # Strip whitespace from OCR output
    # extracted_data = extract_measurements(text, measurement_patterns)
    # print("Extracted measurements:", extracted_data)

    # # --- Test with a more complex string (optional, for robustness check) ---
    # text_complex = "Item A: 10.5 cm, Item B: 25KG, Item C: 100 mm, Temp: 36.6 C and also 98.2F. Weight 2.5lb"
    # print("\n--- Testing with complex string ---")
    # print("Input complex text:", text_complex)
    # extracted_data_complex = extract_measurements(text_complex, measurement_patterns)
    # print("Extracted from complex string:", extracted_data_complex)
    # # --- End of optional test ---

    # Call process_images_from_folder
    target_folder = 'sample_images/'
    if os.path.exists(target_folder):
        logging.info(f"Starting to process folder: {target_folder}")
        process_images_from_folder(target_folder)
        logging.info(f"Finished processing folder: {target_folder}")
    else:
        logging.error(f"Error: Folder '{target_folder}' not found.")