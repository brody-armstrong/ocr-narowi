import pytest
from src.processing.pattern_matcher import PatternMatcher, ReadingType, MedicalReading

def test_pattern_matcher_initialization():
    """Test pattern matcher initialization."""
    matcher = PatternMatcher()
    assert matcher.compiled_patterns is not None
    assert len(matcher.compiled_patterns) == len(ReadingType) - 1  # Excluding UNKNOWN

def test_blood_pressure_detection():
    """Test blood pressure pattern detection."""
    matcher = PatternMatcher()
    test_cases = [
        ("120/80", 95.0),
        ("BP: 140-90", 90.0),
        ("SYS: 130 DIA: 85", 85.0),
        ("Invalid: 300/50", 80.0),  # Should be detected but marked invalid
    ]
    
    for text, confidence in test_cases:
        readings = matcher.find_readings(text, confidence)
        assert len(readings) == 2  # Both systolic and diastolic
        assert all(r.type == ReadingType.BLOOD_PRESSURE for r in readings)
        assert all(r.unit == 'mmHg' for r in readings)
        assert all(r.confidence == confidence for r in readings)
        
        # Check if readings are valid based on ranges
        if "Invalid" in text:
            assert not all(r.is_valid for r in readings)
        else:
            assert all(r.is_valid for r in readings)

def test_temperature_detection():
    """Test temperature pattern detection."""
    matcher = PatternMatcher()
    test_cases = [
        ("98.6F", 95.0),
        ("TEMP: 37.0C", 90.0),
        ("99.5°F", 85.0),
        ("Invalid: 150F", 80.0),  # Should be detected but marked invalid
    ]
    
    for text, confidence in test_cases:
        readings = matcher.find_readings(text, confidence)
        assert len(readings) == 1
        assert readings[0].type == ReadingType.TEMPERATURE
        assert readings[0].confidence == confidence
        
        # Check if reading is valid based on ranges
        if "Invalid" in text:
            assert not readings[0].is_valid
        else:
            assert readings[0].is_valid

def test_weight_detection():
    """Test weight pattern detection."""
    matcher = PatternMatcher()
    test_cases = [
        ("150.5 lbs", 95.0),
        ("WT: 68.2 kg", 90.0),
        ("200.0lb", 85.0),
        ("Invalid: 1000 lbs", 80.0),  # Should be detected but marked invalid
    ]
    
    for text, confidence in test_cases:
        readings = matcher.find_readings(text, confidence)
        assert len(readings) == 1
        assert readings[0].type == ReadingType.WEIGHT
        assert readings[0].confidence == confidence
        
        # Check if reading is valid based on ranges
        if "Invalid" in text:
            assert not readings[0].is_valid
        else:
            assert readings[0].is_valid

def test_oxygen_detection():
    """Test oxygen saturation pattern detection."""
    matcher = PatternMatcher()
    test_cases = [
        ("98%", 95.0),
        ("SPO2: 95", 90.0),
        ("O2: 99%", 85.0),
        ("Invalid: 50%", 80.0),  # Should be detected but marked invalid
    ]
    
    for text, confidence in test_cases:
        readings = matcher.find_readings(text, confidence)
        assert len(readings) == 1
        assert readings[0].type == ReadingType.OXYGEN
        assert readings[0].confidence == confidence
        
        # Check if reading is valid based on ranges
        if "Invalid" in text:
            assert not readings[0].is_valid
        else:
            assert readings[0].is_valid

def test_heart_rate_detection():
    """Test heart rate pattern detection."""
    matcher = PatternMatcher()
    test_cases = [
        ("72 BPM", 95.0),
        ("HR: 85", 90.0),
        ("PULSE: 65 BPM", 85.0),
        ("Invalid: 300 BPM", 80.0),  # Should be detected but marked invalid
    ]
    
    for text, confidence in test_cases:
        readings = matcher.find_readings(text, confidence)
        assert len(readings) == 1
        assert readings[0].type == ReadingType.HEART_RATE
        assert readings[0].confidence == confidence
        
        # Check if reading is valid based on ranges
        if "Invalid" in text:
            assert not readings[0].is_valid
        else:
            assert readings[0].is_valid

def test_unit_conversion():
    """Test unit conversion functionality."""
    matcher = PatternMatcher()
    
    # Test temperature conversion
    temp_f = MedicalReading(
        type=ReadingType.TEMPERATURE,
        value=98.6,
        unit='°F',
        confidence=95.0,
        raw_text="98.6F",
        is_valid=True
    )
    
    temp_c = matcher.convert_unit(temp_f, '°C')
    assert temp_c is not None
    assert temp_c.unit == '°C'
    assert abs(temp_c.value - 37.0) < 0.1
    
    # Test weight conversion
    weight_lbs = MedicalReading(
        type=ReadingType.WEIGHT,
        value=150.0,
        unit='lbs',
        confidence=95.0,
        raw_text="150.0 lbs",
        is_valid=True
    )
    
    weight_kg = matcher.convert_unit(weight_lbs, 'kg')
    assert weight_kg is not None
    assert weight_kg.unit == 'kg'
    assert abs(weight_kg.value - 68.0) < 0.1
    
    # Test invalid conversion
    invalid_conv = matcher.convert_unit(temp_f, 'invalid_unit')
    assert invalid_conv is None

def test_multiple_readings():
    """Test detection of multiple readings in the same text."""
    matcher = PatternMatcher()
    text = "BP: 120/80 TEMP: 98.6F HR: 72 BPM"
    confidence = 95.0
    
    readings = matcher.find_readings(text, confidence)
    assert len(readings) == 4  # 2 for BP, 1 for temp, 1 for HR
    
    # Check that all readings are valid
    assert all(r.is_valid for r in readings)
    
    # Check that we have one of each type
    types = {r.type for r in readings}
    assert ReadingType.BLOOD_PRESSURE in types
    assert ReadingType.TEMPERATURE in types
    assert ReadingType.HEART_RATE in types

def test_invalid_input():
    """Test handling of invalid input."""
    matcher = PatternMatcher()
    
    # Test with empty string
    readings = matcher.find_readings("", 95.0)
    assert len(readings) == 0
    
    # Test with None
    readings = matcher.find_readings(None, 95.0)
    assert len(readings) == 0
    
    # Test with non-numeric text
    readings = matcher.find_readings("No readings here", 95.0)
    assert len(readings) == 0 