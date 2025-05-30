import pytest
import numpy as np
import cv2
from src.processing.image_processor import ImageProcessor
from src.processing.ocr_engine import OCREngine
from src.processing.number_extractor import NumberExtractor, ExtractedNumber

def create_test_image():
    """Create a test image with medical readings."""
    # Create a white background (3-channel BGR image)
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Add some text
    cv2.putText(img, "BP: 120/80", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "Temp: 98.6F", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "O2: 98%", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return img

def test_image_processor():
    """Test image processing functions."""
    processor = ImageProcessor()
    
    # Create test image
    test_img = create_test_image()
    
    # Test validation
    assert processor.validate_image(test_img) == True
    assert processor.validate_image(None) == False
    
    # Test resize
    resized = processor.resize_image(test_img, max_dimension=300)
    assert resized.shape[0] <= 300 or resized.shape[1] <= 300
    
    # Test preprocessing
    preprocessed = processor.preprocess_for_ocr(test_img)
    # Check that the output is grayscale (2D) and has the same height and width
    assert len(preprocessed.shape) == 2
    assert preprocessed.shape[0] == test_img.shape[0]
    assert preprocessed.shape[1] == test_img.shape[1]
    assert preprocessed.dtype == np.uint8

def test_ocr_engine():
    """Test OCR functionality."""
    engine = OCREngine()
    
    # Create test image
    test_img = create_test_image()
    
    # Test text extraction
    text = engine.extract_text(test_img)
    assert isinstance(text, str)
    
    # Test confidence
    confidence = engine.get_confidence(test_img)
    assert 0 <= confidence <= 100
    
    # Test combined extraction
    text, conf = engine.extract_with_confidence(test_img)
    assert isinstance(text, str)
    assert 0 <= conf <= 100

def test_number_extractor():
    """Test number extraction functionality."""
    extractor = NumberExtractor()
    
    # Test blood pressure extraction
    bp_text = "BP: 120/80"
    numbers = extractor.extract_numbers(bp_text, confidence=90.0)
    assert len(numbers) == 2
    assert numbers[0].value == 120
    assert numbers[1].value == 80
    assert numbers[0].unit == 'mmHg'
    
    # Test temperature extraction
    temp_text = "Temp: 98.6F"
    numbers = extractor.extract_numbers(temp_text, confidence=90.0)
    assert len(numbers) == 1
    assert numbers[0].value == 98.6
    assert numbers[0].unit == '°F'
    
    # Test validation
    valid_bp = ExtractedNumber(value=120, unit='mmHg', confidence=90.0, raw_text="120/80")
    invalid_bp = ExtractedNumber(value=300, unit='mmHg', confidence=90.0, raw_text="300/80")
    assert extractor.validate_reading(valid_bp) == True
    assert extractor.validate_reading(invalid_bp) == False

def test_integration():
    """Test integration of all components."""
    # Create test image
    test_img = create_test_image()
    
    # Process image
    processor = ImageProcessor()
    preprocessed = processor.preprocess_for_ocr(test_img)
    
    # Perform OCR
    engine = OCREngine()
    text, confidence = engine.extract_with_confidence(preprocessed)
    
    # Extract numbers
    extractor = NumberExtractor()
    numbers = extractor.extract_numbers(text, confidence)
    
    # Verify results
    assert len(numbers) > 0
    for number in numbers:
        assert isinstance(number, ExtractedNumber)
        assert extractor.validate_reading(number) 