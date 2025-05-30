import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ReadingType(Enum):
    """Types of medical readings that can be detected."""
    BLOOD_PRESSURE = "blood_pressure"
    TEMPERATURE = "temperature"
    WEIGHT = "weight"
    OXYGEN = "oxygen"
    HEART_RATE = "heart_rate"
    UNKNOWN = "unknown"

@dataclass
class MedicalReading:
    """Class to hold information about a detected medical reading."""
    type: ReadingType
    value: float
    unit: str
    confidence: float
    raw_text: str
    is_valid: bool = True

class PatternMatcher:
    """Identifies and categorizes medical readings from text."""
    # Patterns are ordered from most specific to least specific
    PATTERNS = [
        # Blood pressure (must come first)
        (ReadingType.BLOOD_PRESSURE, re.compile(r'(?:^|\s)(?:BP:)?\s*(\d{2,3})[/-](\d{2,3})(?:\s|$)', re.IGNORECASE), 'mmHg'),
        (ReadingType.BLOOD_PRESSURE, re.compile(r'(?:^|\s)SYS:\s*(\d{2,3})\s*DIA:\s*(\d{2,3})(?:\s|$)', re.IGNORECASE), 'mmHg'),
        # Temperature (more flexible with decimal places and spacing)
        (ReadingType.TEMPERATURE, re.compile(r'(?:^|\s)(?:TEMP:)?\s*(\d{2,3}(?:\.\d{1,2})?)\s*[°]?[Ff](?=\b|\s|$)', re.IGNORECASE), '°F'),
        (ReadingType.TEMPERATURE, re.compile(r'(?:^|\s)(?:TEMP:)?\s*(\d{2,3}(?:\.\d{1,2})?)\s*[°]?[Cc](?=\b|\s|$)', re.IGNORECASE), '°C'),
        # Weight (match anywhere in the string)
        (ReadingType.WEIGHT, re.compile(r'(?:WT:)?\s*(\d{2,3}(?:\.\d{1,2})?)\s*(lbs?|lb)(?=\b|\s|$)', re.IGNORECASE), 'lbs'),
        (ReadingType.WEIGHT, re.compile(r'(?:WT:)?\s*(\d{2,3}(?:\.\d{1,2})?)\s*(kg|kgs)(?=\b|\s|$)', re.IGNORECASE), 'kg'),
        # Heart rate: require HR, PULSE, or BPM as prefix or suffix (not just a number)
        (ReadingType.HEART_RATE, re.compile(r'(?:^|\s)(?:HR:|PULSE:)?\s*(\d{2,3})\s*(BPM|HR)(?:\s|$)', re.IGNORECASE), 'BPM'),
        (ReadingType.HEART_RATE, re.compile(r'(?:^|\s)HR:\s*(\d{2,3})(?:\s|$)', re.IGNORECASE), 'BPM'),
        (ReadingType.HEART_RATE, re.compile(r'(?:^|\s)PULSE:\s*(\d{2,3})(?:\s|$)', re.IGNORECASE), 'BPM'),
        # Oxygen: require SPO2, O2, or % (not just a number)
        (ReadingType.OXYGEN, re.compile(r'(?:^|\s)(?:SPO2:|O2:)?\s*(\d{2,3})\s*%(?:\s|$)', re.IGNORECASE), '%'),
        (ReadingType.OXYGEN, re.compile(r'(?:^|\s)(SPO2:|O2:)\s*(\d{2,3})(?:\s|$)', re.IGNORECASE), '%'),
    ]

    VALID_RANGES = {
        ReadingType.BLOOD_PRESSURE: {
            'systolic': (60, 200),
            'diastolic': (40, 120)
        },
        ReadingType.TEMPERATURE: {
            '°F': (95, 105),
            '°C': (35, 41)
        },
        ReadingType.WEIGHT: {
            'lbs': (50, 500),
            'kg': (20, 250)
        },
        ReadingType.OXYGEN: {
            '%': (70, 100)
        },
        ReadingType.HEART_RATE: {
            'BPM': (40, 200)
        }
    }

    def __init__(self):
        # For backward compatibility with tests
        self.compiled_patterns = {
            reading_type: [(p.pattern, unit) for rt, p, unit in self.PATTERNS if rt == reading_type]
            for reading_type in ReadingType if reading_type != ReadingType.UNKNOWN
        }

    def find_readings(self, text: str, confidence: float) -> List[MedicalReading]:
        if text is None:
            return []
        readings = []
        used_ranges = []  # list of (start, end) tuples
        for reading_type, pattern, unit in self.PATTERNS:
            for match in pattern.finditer(text):
                start, end = match.span()
                match_text = match.group(0)
                # Calculate the span of the match after stripping leading/trailing whitespace
                ws_prefix = len(match_text) - len(match_text.lstrip())
                ws_suffix = len(match_text) - len(match_text.rstrip())
                value_start = start + ws_prefix
                value_end = end - ws_suffix
                # Skip if this overlaps with any already-used range (using value span)
                if any(not (value_end <= s or value_start >= e) for s, e in used_ranges):
                    continue
                if reading_type == ReadingType.BLOOD_PRESSURE:
                    systolic = float(match.group(1))
                    diastolic = float(match.group(2))
                    readings.append(MedicalReading(
                        type=reading_type,
                        value=systolic,
                        unit=unit,
                        confidence=confidence,
                        raw_text=match.group(0).strip(),
                        is_valid=self._validate_reading(reading_type, systolic, unit, 'systolic')
                    ))
                    readings.append(MedicalReading(
                        type=reading_type,
                        value=diastolic,
                        unit=unit,
                        confidence=confidence,
                        raw_text=match.group(0).strip(),
                        is_valid=self._validate_reading(reading_type, diastolic, unit, 'diastolic')
                    ))
                elif reading_type == ReadingType.OXYGEN and len(match.groups()) == 2:
                    # For the (SPO2:|O2:)\s*(\d{2,3}) pattern
                    value = float(match.group(2))
                    readings.append(MedicalReading(
                        type=reading_type,
                        value=value,
                        unit=unit,
                        confidence=confidence,
                        raw_text=match.group(0).strip(),
                        is_valid=self._validate_reading(reading_type, value, unit)
                    ))
                else:
                    value = float(match.group(1))
                    readings.append(MedicalReading(
                        type=reading_type,
                        value=value,
                        unit=unit,
                        confidence=confidence,
                        raw_text=match.group(0).strip(),
                        is_valid=self._validate_reading(reading_type, value, unit)
                    ))
                used_ranges.append((value_start, value_end))
        return readings

    def _validate_reading(self, 
                         reading_type: ReadingType, 
                         value: float, 
                         unit: str,
                         bp_type: Optional[str] = None) -> bool:
        if reading_type not in self.VALID_RANGES:
            return True
        ranges = self.VALID_RANGES[reading_type]
        if reading_type == ReadingType.BLOOD_PRESSURE and bp_type:
            min_val, max_val = ranges[bp_type]
        else:
            min_val, max_val = ranges[unit]
        return min_val <= value <= max_val

    def convert_unit(self, reading: MedicalReading, target_unit: str) -> Optional[MedicalReading]:
        if reading.type == ReadingType.TEMPERATURE:
            if reading.unit == '°F' and target_unit == '°C':
                new_value = (reading.value - 32) * 5/9
                return MedicalReading(
                    type=reading.type,
                    value=round(new_value, 1),
                    unit=target_unit,
                    confidence=reading.confidence,
                    raw_text=reading.raw_text,
                    is_valid=reading.is_valid
                )
            elif reading.unit == '°C' and target_unit == '°F':
                new_value = (reading.value * 9/5) + 32
                return MedicalReading(
                    type=reading.type,
                    value=round(new_value, 1),
                    unit=target_unit,
                    confidence=reading.confidence,
                    raw_text=reading.raw_text,
                    is_valid=reading.is_valid
                )
        elif reading.type == ReadingType.WEIGHT:
            if reading.unit == 'lbs' and target_unit == 'kg':
                new_value = reading.value * 0.453592
                return MedicalReading(
                    type=reading.type,
                    value=round(new_value, 1),
                    unit=target_unit,
                    confidence=reading.confidence,
                    raw_text=reading.raw_text,
                    is_valid=reading.is_valid
                )
            elif reading.unit == 'kg' and target_unit == 'lbs':
                new_value = reading.value * 2.20462
                return MedicalReading(
                    type=reading.type,
                    value=round(new_value, 1),
                    unit=target_unit,
                    confidence=reading.confidence,
                    raw_text=reading.raw_text,
                    is_valid=reading.is_valid
                )
        return None 