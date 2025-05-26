import cv2
import pytesseract
import os
import argparse
import logging

def process_images_from_folder(folder_path):

    custom_config = r'--oem 3 --psm 6 outputbase digits' #treats the image as a single character

    if not os.path.exists(folder_path):
        logging.error(f"Folder not found at {folder_path}")
        return
        
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
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                               cv2.THRESH_BINARY_INV, 11, 2)
                countours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                rois_with_text = []

                for i, countour in enumerate(countours):
                    x, y, w, h = cv2.boundingRect(countour)
                    if w > 10 and h > 10 and w < thresh.shape[1]:
                        roi = thresh[y:y+h, x:x+w]

                        try:

                            text_roi = pytesseract.image_to_string(roi, config=custom_config).strip()
                            if text_roi:
                                rois_with_text.append(((x, y, w, h), text_roi))
                        except pytesseract.TesseractError as e:
                            logging.error(f"Tesseract OCR failed for an ROI in {filename} (ROI #{i+1}): {e}")
                        except Exception as e: # Catch other potential errors during OCR
                            logging.error(f"An unexpected error occurred during OCR for an ROI in {filename} (ROI #{i+1}): {e}")
                sorted_rois = sorted(rois_with_text, key = lambda item: (item[0][1], item[0][0]))

                text = " ".join([item[1] for item in sorted_rois]) 
                logging.info(f"Detected text in {filename}: {text}")
                if not text:
                    logging.warning(f"No text detected in {filename}.")
                else:
                    print(f"\n🧾 OCR result for {filename}:\n{text}\n")
                
                for (bbox, roi_text) in sorted_rois:
                    x, y, w, h = bbox
                    print(f"  ROI at ({x},{y},{w},{h}) → '{roi_text}'")

            except Exception as e:
                logging.error(f"Failed to process image {filename}: {e}")       
        else:
            if os.path.isfile(img_path): 
                 logging.warning(f"Skipping {filename}: not a recognized image file type (png, jpg, jpeg).")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Process images in a folder to extract text using OCR.")
    parser.add_argument("folder_path", type=str, help="Path to the folder containing images.")
    
    try:
        args = parser.parse_args()
        process_images_from_folder(args.folder_path)
    except SystemExit:
        # Argparse can cause SystemExit (e.g., for --help). This is normal.
        logging.info("Application exited via argparse (e.g., help displayed or argument error).")
    except Exception as e:
        logging.critical(f"An unexpected critical error occurred in the main application flow: {e}")

if __name__ == "__main__":
    main()
