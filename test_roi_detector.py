import pytest
import numpy as np
import cv2
from src.processing.roi_detector import ROIDetector, DisplayRegion

def create_test_image():
    """Create a test image with simulated medical device displays."""
    # Create a white background
    img = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw main display (LCD-like): uniform mid-gray fill, thick black border
    cv2.rectangle(img, (100, 100), (400, 300), (160, 160, 160), -1)  # LCD fill
    cv2.rectangle(img, (100, 100), (400, 300), (0, 0, 0), 6)         # LCD border
    cv2.putText(img, "120/80", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    
    # Draw secondary display: lighter fill, thinner border
    cv2.rectangle(img, (450, 100), (700, 200), (210, 210, 210), -1)
    cv2.rectangle(img, (450, 100), (700, 200), (0, 0, 0), 2)
    cv2.putText(img, "98.6F", (470, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Add some noise/artifacts
    noise = np.random.normal(0, 8, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img

def test_roi_detector_initialization():
    """Test ROI detector initialization with custom parameters."""
    detector = ROIDetector(
        min_area=500,
        max_area=50000,
        aspect_ratio_range=(0.2, 3.0)
    )
    
    assert detector.min_area == 500
    assert detector.max_area == 50000
    assert detector.aspect_ratio_range == (0.2, 3.0)

def test_display_detection():
    """Test basic display detection functionality."""
    detector = ROIDetector()
    test_img = create_test_image()
    
    # Detect displays
    regions = detector.detect_displays(test_img)
    
    # Should detect at least the main displays
    assert len(regions) >= 2
    
    # Check region properties
    for region in regions:
        assert isinstance(region, DisplayRegion)
        assert 0 <= region.confidence <= 100
        assert region.width > 0 and region.height > 0
        assert region.x >= 0 and region.y >= 0

def test_lcd_detection():
    """Test LCD display detection."""
    detector = ROIDetector()
    test_img = create_test_image()
    
    regions = detector.detect_displays(test_img)
    
    # Should identify at least one LCD display
    lcd_regions = [r for r in regions if r.is_lcd]
    assert len(lcd_regions) > 0
    
    # Check LCD region properties
    for region in lcd_regions:
        assert region.confidence > 50  # LCD regions should have high confidence

def test_region_visualization():
    """Test region visualization functionality."""
    detector = ROIDetector()
    test_img = create_test_image()
    
    # Detect and visualize regions
    regions = detector.detect_displays(test_img)
    visualized = detector.draw_regions(test_img, regions)
    
    # Check visualization properties
    assert visualized.shape == test_img.shape
    assert visualized.dtype == test_img.dtype
    
    # Check that visualization is different from original
    assert not np.array_equal(visualized, test_img)

def test_confidence_calculation():
    """Test confidence score calculation."""
    detector = ROIDetector()
    test_img = create_test_image()
    
    regions = detector.detect_displays(test_img)
    
    # Check confidence scores
    for region in regions:
        assert 0 <= region.confidence <= 100
        
        # Regions with good aspect ratio should have higher confidence
        aspect_ratio = region.width / region.height
        if 0.8 <= aspect_ratio <= 1.2:
            assert region.confidence > 50

def test_invalid_input():
    """Test handling of invalid input."""
    detector = ROIDetector()
    
    # Test with empty image
    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    regions = detector.detect_displays(empty_img)
    assert len(regions) == 0
    
    # Test with None
    regions = detector.detect_displays(None)
    assert len(regions) == 0 