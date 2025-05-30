import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional

class ImageProcessor:
    """Handles basic image preprocessing for OCR."""
    
    @staticmethod
    def load_image(image_path: str) -> Optional[np.ndarray]:
        """
        Load an image from the given path.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            numpy.ndarray: Loaded image or None if loading fails
        """
        try:
            return cv2.imread(image_path)
        except Exception as e:
            print(f"Error loading image: {e}")
            return None

    @staticmethod
    def validate_image(image: np.ndarray) -> bool:
        """
        Validate if the image is suitable for processing.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            bool: True if image is valid, False otherwise
        """
        if image is None:
            return False
        if image.size == 0:
            return False
        return True

    @staticmethod
    def resize_image(image: np.ndarray, max_dimension: int = 2000) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            max_dimension: Maximum dimension (width or height)
            
        Returns:
            numpy.ndarray: Resized image
        """
        height, width = image.shape[:2]
        scale = min(max_dimension / width, max_dimension / height)
        if scale < 1:
            new_width = int(width * scale)
            new_height = int(height * scale)
            return cv2.resize(image, (new_width, new_height))
        return image

    @staticmethod
    def adjust_contrast_brightness(image: np.ndarray, 
                                 contrast: float = 1.2, 
                                 brightness: int = 10) -> np.ndarray:
        """
        Adjust image contrast and brightness.
        
        Args:
            image: Input image
            contrast: Contrast adjustment factor
            brightness: Brightness adjustment value
            
        Returns:
            numpy.ndarray: Adjusted image
        """
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted

    @staticmethod
    def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Apply basic preprocessing pipeline for OCR.
        
        Args:
            image: Input image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh

    @staticmethod
    def save_image(image: np.ndarray, output_path: str) -> bool:
        """
        Save processed image to file.
        
        Args:
            image: Image to save
            output_path: Path where image should be saved
            
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            cv2.imwrite(output_path, image)
            return True
        except Exception as e:
            print(f"Error saving image: {e}")
            return False 