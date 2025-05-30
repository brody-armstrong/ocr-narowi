import pytesseract
from typing import Dict, Optional, Tuple
import cv2
import numpy as np

class OCREngine:
    """Wrapper for Tesseract OCR functionality."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize OCR engine with optional configuration.
        
        Args:
            config: Dictionary of Tesseract configuration parameters
        """
        self.config = config or {
            '--oem': '1',  # Use Legacy + LSTM OCR Engine Mode
            '--psm': '6',  # Assume uniform block of text
            'tessedit_char_whitelist': '0123456789./-',  # Only look for numbers and basic symbols
        }
        
    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from preprocessed image.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            str: Extracted text
        """
        try:
            # Build config string for Tesseract
            config_str = ''
            for k, v in self.config.items():
                if k.startswith('--'):
                    config_str += f'{k} {v} '
                else:
                    config_str += f'-c {k}={v} '
            config_str = config_str.strip()
            
            # Perform OCR
            text = pytesseract.image_to_string(
                image,
                config=config_str
            )
            return text.strip()
        except Exception as e:
            print(f"Error during OCR: {e}")
            return ""

    def get_confidence(self, image: np.ndarray) -> float:
        """
        Get confidence score for OCR result.
        
        Args:
            image: Preprocessed image
            
        Returns:
            float: Confidence score (0-100)
        """
        try:
            config_str = ''
            for k, v in self.config.items():
                if k.startswith('--'):
                    config_str += f'{k} {v} '
                else:
                    config_str += f'-c {k}={v} '
            config_str = config_str.strip()
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=config_str
            )
            
            # Calculate average confidence for non-empty text
            confidences = [int(conf) for conf, text in zip(data['conf'], data['text']) 
                         if text.strip()]
            return sum(confidences) / len(confidences) if confidences else 0.0
        except Exception as e:
            print(f"Error getting confidence: {e}")
            return 0.0

    def extract_with_confidence(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Extract text and confidence score from image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Tuple[str, float]: (extracted text, confidence score)
        """
        text = self.extract_text(image)
        confidence = self.get_confidence(image)
        return text, confidence 