import pytesseract
from typing import Dict, Optional, Tuple
import numpy as np

class OCREngine:
    """
    Wrapper for Tesseract OCR functionality.

    Common Tesseract Page Segmentation Modes (PSM) (--psm option):
      0    Orientation and script detection (OSD) only.
      1    Automatic page segmentation with OSD.
      2    Automatic page segmentation, but no OSD, or OCR.
      3    Fully automatic page segmentation, but no OSD. (Default)
      4    Assume a single column of text of variable sizes.
      5    Assume a single uniform block of vertically aligned text.
      6    Assume a single uniform block of text. (Used as default in this class)
      7    Treat the image as a single text line.
      8    Treat the image as a single word.
      9    Treat the image as a single word in a circle.
     10    Treat the image as a single character.
     11    Sparse text. Find as much text as possible in no particular order.
     12    Sparse text with OSD.
     13    Raw line. Treat the image as a single text line,
           bypassing hacks that are Tesseract-specific.
    """
    
    def __init__(self, config: Optional[Dict[str, str]] = None):
        """
        Initialize OCR engine with optional Tesseract configuration.
        
        The PSM (Page Segmentation Mode) can be set via the `config` dictionary.
        If `config` is provided and contains a `'--psm'` key, its value will be used.
        Otherwise, it defaults to '6'.

        Args:
            config: Dictionary of Tesseract configuration parameters. 
                    Example: `{'--psm': '7', 'tessedit_char_whitelist': '0123456789'}`
        """
        default_config = {
            '--oem': '1',  # Use Legacy + LSTM OCR Engine Mode
            '--psm': '6',  # Assume uniform block of text (default for this class)
            'tessedit_char_whitelist': '0123456789./-',  # Only look for numbers and basic symbols
        }
        
        if config:
            # Merge provided config with defaults, provided config takes precedence
            self.config = {**default_config, **config}
        else:
            self.config = default_config
        
        print(f"OCREngine initialized with config: {self.config}")

    def set_psm(self, psm_value: str):
        """
        Update the Page Segmentation Mode (PSM) for Tesseract.

        Args:
            psm_value: The new PSM value (e.g., '3', '6', '7').
                       Refer to class docstring for common PSM values.
        """
        if not isinstance(psm_value, str):
            print(f"Warning: PSM value should be a string. Attempting to use {psm_value} as string.")
            psm_value = str(psm_value)
            
        self.config['--psm'] = psm_value
        print(f"Tesseract PSM updated to: {psm_value}. Current config: {self.config}")

    def _build_config_str(self) -> str:
        """Helper method to build the Tesseract config string."""
        config_str_parts = []
        for k, v in self.config.items():
            if k.startswith('--'):
                config_str_parts.append(f'{k} {v}')
            else:
                config_str_parts.append(f'-c {k}={v}')
        return ' '.join(config_str_parts)

    def extract_text(self, image: np.ndarray) -> str:
        """
        Extract text from preprocessed image.
        
        Args:
            image: Preprocessed image as numpy array
            
        Returns:
            str: Extracted text
        """
        try:
            config_str = self._build_config_str()
            print(f"Using Tesseract config for extract_text: '{config_str}'") # Logging added
            
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
            config_str = self._build_config_str()
            # Not printing config_str here to avoid too much noise if called frequently with extract_with_confidence
            data = pytesseract.image_to_data(
                image,
                output_type=pytesseract.Output.DICT,
                config=config_str
            )
            
            # Calculate average confidence for non-empty text, ignoring negative confidences
            confidences = [int(conf) for conf, text in zip(data['conf'], data['text']) 
                         if text.strip() and int(conf) >= 0]
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