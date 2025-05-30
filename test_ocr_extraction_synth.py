import os
import pytest
import cv2
import numpy as np
from src.processing.image_processor import ImageProcessor
from src.processing.ocr_engine import OCREngine

def test_ocr_extraction_synth():
    """Test OCR extraction from synthetic thermometer images."""
    # Paths to the synthetic images
    image_paths = [
        'thermometer_synth_1.png',
        'thermometer_synth_2.png',
        'thermometer_synth_3.png'
    ]

    # Initialize the image processor and OCR engine
    processor = ImageProcessor()
    engine = OCREngine()

    for image_path in image_paths:
        # Check if the image file exists
        assert os.path.exists(image_path), f"Test image not found at {image_path}"

        # Load and preprocess the image
        image = processor.load_image(image_path)
        assert image is not None, f"Failed to load image: {image_path}"

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Apply Otsu's thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Apply morphological operations to clean up the image
        kernel = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # Extract text from the preprocessed image
        text, confidence = engine.extract_with_confidence(cleaned)

        # Print debug information
        print(f"\nImage: {image_path}")
        print(f"Image dimensions: {image.shape}")
        print(f"Extracted Text: {text}")
        print(f"Confidence: {confidence}")

        # Assert that text was extracted
        assert text, f"No text was extracted from the image: {image_path}"

        # Optionally, assert that the confidence is above a certain threshold
        assert confidence > 0.5, f"OCR confidence is too low for image: {image_path}"