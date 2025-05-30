import re
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class ExtractedNumber:
    """Class to hold extracted number information."""
    value: float
    unit: Optional[str]
    confidence: float
    raw_text: str

class NumberExtractor:
    """Extracts numerical values from OCR text."""
    
    # Common patterns for medical readings
    PATTERNS = {
        'blood_pressure': r'(\d{2,3})[/-](\d{2,3})',  # 120/80 or 120-80
        'temperature': r'(\d{2,3}\.\d{1,2})[°]?[FC]',  # 98.6°F or 37.0°C
        'weight': r'(\d{2,3}\.\d{1,2})\s*(?:lbs|kg)',  # 150.5 lbs or 68.2 kg
        'oxygen': r'(\d{2,3})%',  # 98%
        'heart_rate': r'(\d{2,3})\s*(?:BPM|HR)',  # 72 BPM or HR: 72
    }
    
    def __init__(self):
        """Initialize number extractor with compiled regex patterns."""
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }
    
    def extract_numbers(self, text: str, confidence: float) -> List[ExtractedNumber]:
        """
        Extract all numerical values from text.
        
        Args:
            text: OCR extracted text
            confidence: OCR confidence score
            
        Returns:
            List[ExtractedNumber]: List of extracted numbers with metadata
        """
        results = []
        
        # Try each pattern
        for reading_type, pattern in self.compiled_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                if reading_type == 'blood_pressure':
                    # Handle blood pressure as two numbers
                    systolic = float(match.group(1))
                    diastolic = float(match.group(2))
                    results.extend([
                        ExtractedNumber(
                            value=systolic,
                            unit='mmHg',
                            confidence=confidence,
                            raw_text=match.group(0)
                        ),
                        ExtractedNumber(
                            value=diastolic,
                            unit='mmHg',
                            confidence=confidence,
                            raw_text=match.group(0)
                        )
                    ])
                else:
                    # Handle single number readings
                    value = float(match.group(1))
                    unit = self._get_unit(reading_type, match.group(0))
                    results.append(
                        ExtractedNumber(
                            value=value,
                            unit=unit,
                            confidence=confidence,
                            raw_text=match.group(0)
                        )
                    )
        
        return results
    
    def _get_unit(self, reading_type: str, raw_text: str) -> Optional[str]:
        """
        Determine the unit for a reading type.
        
        Args:
            reading_type: Type of medical reading
            raw_text: Raw text containing the reading
            
        Returns:
            Optional[str]: Unit string or None
        """
        unit_map = {
            'temperature': '°F' if 'F' in raw_text.upper() else '°C',
            'weight': 'lbs' if 'lbs' in raw_text.lower() else 'kg',
            'oxygen': '%',
            'heart_rate': 'BPM'
        }
        return unit_map.get(reading_type)
    
    def validate_reading(self, number: ExtractedNumber) -> bool:
        """
        Validate if the extracted number is within reasonable medical ranges.
        
        Args:
            number: Extracted number to validate
            
        Returns:
            bool: True if reading is within valid range
        """
        # Basic range validations
        ranges = {
            'mmHg': (60, 200),  # Blood pressure
            '°F': (95, 105),    # Temperature
            '°C': (35, 41),     # Temperature
            'lbs': (50, 500),   # Weight
            'kg': (20, 250),    # Weight
            '%': (70, 100),     # Oxygen saturation
            'BPM': (40, 200)    # Heart rate
        }
        
        if number.unit in ranges:
            min_val, max_val = ranges[number.unit]
            return min_val <= number.value <= max_val
        
        return True  # Unknown units are considered valid 