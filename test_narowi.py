import unittest
import tempfile
import os
import shutil
import cv2
import numpy as np
import logging
import io

# Attempt to import the script to be tested
# Ensure narowi.py is in the same directory or accessible via PYTHONPATH
import narowi 

class TestNarowiFunctionality(unittest.TestCase):

    def setUp(self):
        # The setUp for creating temporary image files might still be useful for future end-to-end tests.
        # For now, it's not strictly needed for the unit tests being added.
        self.test_dir = tempfile.mkdtemp()
        # self.test_image_name = "test_ocr_image.png"
        # self.test_image_path = os.path.join(self.test_dir, self.test_image_name)
        # self.create_test_image(self.test_image_path, "123 789") # Example image creation

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    # def create_test_image(self, file_path, text_to_write, width=450, height=100):
    #     """
    #     Creates an image with specified text. (Helper for potential future tests)
    #     """
    #     image = np.ones((height, width, 3), dtype=np.uint8) * 255 # White background
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     font_scale = 2
    #     font_color = (0, 0, 0) # Black
    #     thickness = 3
    #     text_size = cv2.getTextSize(text_to_write, font, font_scale, thickness)[0]
    #     text_x = (width - text_size[0]) // 2
    #     text_y = (height + text_size[1]) // 2
    #     cv2.putText(image, text_to_write, (text_x, text_y), font, font_scale, font_color, thickness)
    #     cv2.imwrite(file_path, image)

    # def test_process_single_image_correct_text(self):
    #     # This test is commented out as it's outdated and relies on log parsing.
    #     # End-to-end tests will be added later with a different approach.
    #     pass

    def test_identify_device_type(self):
        self.assertEqual(narowi.identify_device_type("SYS 120 DIA 80 PUL 60"), "blood_pressure_monitor")
        self.assertEqual(narowi.identify_device_type("Spo2 98% PR 70"), "oximeter")
        self.assertEqual(narowi.identify_device_type("Temp: 36.5 °C"), "thermometer")
        self.assertEqual(narowi.identify_device_type("Weight 70.5 kg"), "weight_scale")
        self.assertEqual(narowi.identify_device_type("Just some random text"), "unknown_device")
        self.assertEqual(narowi.identify_device_type(""), "unknown_device")
        self.assertEqual(narowi.identify_device_type("sys 120 dia 80 pul 60"), "blood_pressure_monitor", "Test case-insensitivity")
        self.assertEqual(narowi.identify_device_type("PULSE 77 DIA 88 SYS 123"), "blood_pressure_monitor", "Test keyword order")

    def test_summarize_image_data(self):
        # Sample data for a blood pressure monitor
        image_data_bp = {
            "image_filename": "bp_test.png",
            "device_type": "blood_pressure_monitor",
            "readings": {"systolic": 120, "diastolic": 80, "pulse": 60},
            # Example error where pulse is out of typical range for this specific (hypothetical) check
            # Note: The actual PHYSIOLOGICAL_RANGES for pulse is 40-180, so 60 is fine.
            # Let's adjust the error to be more illustrative or remove if it contradicts current ranges.
            # For this test, we'll assume the error string is generated as is.
            "errors": ["Warning: Pulse value 60 bpm is outside the typical range of 70-120 bpm."], 
            "confidence_scores": {"systolic": 95, "diastolic": 92, "pulse": 88}
        }
        summary_bp = narowi.summarize_image_data(image_data_bp)
        self.assertIn("--- Summary for bp_test.png ---", summary_bp)
        self.assertIn("Device Type: blood_pressure_monitor", summary_bp)
        self.assertIn("Systolic: 120 mmHg (Confidence: 95%)", summary_bp)
        self.assertIn("Diastolic: 80 mmHg (Confidence: 92%)", summary_bp)
        self.assertIn("Pulse: 60 bpm (Confidence: 88%)", summary_bp) # narowi.PHYSIOLOGICAL_RANGES has 'bpm' for pulse
        self.assertIn("Warning: Pulse value 60 bpm is outside the typical range of 70-120 bpm.", summary_bp)

        # Sample data for an unknown device
        image_data_unknown = {
            "image_filename": "unknown_test.png",
            "device_type": "unknown_device",
            "readings": {}, 
            "errors": [], 
            "confidence_scores": {}
        }
        summary_unknown = narowi.summarize_image_data(image_data_unknown)
        self.assertIn("--- Summary for unknown_test.png ---", summary_unknown)
        self.assertIn("Device Type: unknown_device", summary_unknown)
        self.assertIn("Readings: No readings detected.", summary_unknown)
        self.assertNotIn("Alerts/Errors:", summary_unknown) # No errors, so this section shouldn't appear.

        # Sample data with missing confidence for one reading
        image_data_missing_conf = {
            "image_filename": "missing_conf.png",
            "device_type": "oximeter",
            "readings": {"spo2": 97, "pulse_rate": 75},
            "errors": [],
            "confidence_scores": {"spo2": 91} # Confidence for pulse_rate is missing
        }
        summary_missing_conf = narowi.summarize_image_data(image_data_missing_conf)
        self.assertIn("Spo2: 97 % (Confidence: 91%)", summary_missing_conf) # Spo2 unit is '%'
        self.assertIn("Pulse Rate: 75 bpm (Confidence: N/A)", summary_missing_conf)


    def _create_mock_detailed_roi_item(self, text, confidence, bbox_parts):
        """
        Helper to create a mock item for sorted_detailed_rois.
        bbox_parts: tuple (x, y, w, h)
        """
        return {'text': text, 'confidence': confidence, 'bbox': bbox_parts}

    def test_classify_readings_structure(self):
        """
        Tests the basic return structure of classify_readings.
        More detailed tests for classify_readings will be added later.
        """
        # Example: Using the helper to create mock data if needed, though not for this basic test
        # mock_roi_item = self._create_mock_detailed_roi_item("SYS", 90, (10,10,30,20))
        
        result = narowi.classify_readings([], "unknown_device")
        self.assertIsInstance(result, dict)
        self.assertIn("readings", result)
        self.assertIsInstance(result["readings"], dict)
        self.assertIn("errors", result)
        self.assertIsInstance(result["errors"], list)
        self.assertIn("confidence_scores", result)
        self.assertIsInstance(result["confidence_scores"], dict)

        # Check with a known device type but empty ROIs
        result_bp = narowi.classify_readings([], "blood_pressure_monitor")
        self.assertEqual(result_bp, {"readings": {}, "errors": [], "confidence_scores": {}})

    def test_classify_readings_extraction_and_errors(self):
        # Test Case 1: Simple Number and Label Extraction (Blood Pressure Monitor)
        mock_rois_bp = [
            self._create_mock_detailed_roi_item(text="SYS", confidence=90, bbox_parts=(10,10,30,20)),
            self._create_mock_detailed_roi_item(text="120", confidence=95, bbox_parts=(50,10,40,20)),
            self._create_mock_detailed_roi_item(text="DIA", confidence=88, bbox_parts=(10,35,30,20)),
            self._create_mock_detailed_roi_item(text="80", confidence=92, bbox_parts=(50,35,30,20)),
            self._create_mock_detailed_roi_item(text="PUL", confidence=85, bbox_parts=(10,60,30,20)),
            self._create_mock_detailed_roi_item(text="70", confidence=90, bbox_parts=(50,60,30,20)),
        ]
        result_bp = narowi.classify_readings(mock_rois_bp, "blood_pressure_monitor")
        self.assertEqual(result_bp["errors"], [], "Test Case 1: Errors should be empty for valid BP data.")
        # Potential numbers/labels are internal, so not asserted directly here.

        # Test Case 2: Number Parsing Error
        mock_rois_parse_error = [
            self._create_mock_detailed_roi_item(text="SYS", confidence=90, bbox_parts=(10,10,30,20)),
            self._create_mock_detailed_roi_item(text="12O", confidence=80, bbox_parts=(50,10,40,20)), # 'O' instead of '0'
        ]
        result_parse_error = narowi.classify_readings(mock_rois_parse_error, "blood_pressure_monitor")
        self.assertEqual(len(result_parse_error["errors"]), 1, "Test Case 2: Should be one error for parsing '12O'.")
        self.assertIn("Could not convert extracted number '12O'", result_parse_error["errors"][0])
        self.assertIn("ROI: '12O'", result_parse_error["errors"][0])
        self.assertIn("Conf: 80", result_parse_error["errors"][0])

        # Test Case 3: Decimal Number Extraction (Thermometer)
        mock_rois_temp = [
            self._create_mock_detailed_roi_item(text="TEMP", confidence=90, bbox_parts=(10,10,40,20)),
            self._create_mock_detailed_roi_item(text="37.5", confidence=95, bbox_parts=(60,10,40,20)),
            self._create_mock_detailed_roi_item(text="°C", confidence=88, bbox_parts=(110,10,20,20)),
        ]
        result_temp = narowi.classify_readings(mock_rois_temp, "thermometer")
        self.assertEqual(result_temp["errors"], [], "Test Case 3: Errors should be empty for valid temperature data.")
        # Internal potential_numbers should contain 37.5

        # Test Case 4: Unknown Device Type (using BP data)
        result_unknown_device = narowi.classify_readings(mock_rois_bp, "unknown_device")
        self.assertEqual(result_unknown_device["errors"], [], "Test Case 4: Errors should be empty for unknown device type (no label specific processing).")
        # Numbers should still be extracted to potential_numbers internally, but no labels identified from READING_LABELS

        # Test Case 5: No Numbers, Only Labels
        mock_rois_only_labels = [
            self._create_mock_detailed_roi_item(text="SYS", confidence=90, bbox_parts=(10,10,30,20)),
            self._create_mock_detailed_roi_item(text="DIA", confidence=88, bbox_parts=(10,35,30,20)),
        ]
        result_only_labels = narowi.classify_readings(mock_rois_only_labels, "blood_pressure_monitor")
        self.assertEqual(result_only_labels["errors"], [], "Test Case 5: Errors should be empty when only labels are present.")
        # Internal potential_labels should be populated, potential_numbers empty.

        # Test Case 6: No Labels, Only Numbers
        mock_rois_only_numbers = [
            self._create_mock_detailed_roi_item(text="120", confidence=95, bbox_parts=(50,10,40,20)),
            self._create_mock_detailed_roi_item(text="80", confidence=92, bbox_parts=(50,35,30,20)),
        ]
        result_only_numbers = narowi.classify_readings(mock_rois_only_numbers, "blood_pressure_monitor")
        self.assertEqual(result_only_numbers["errors"], [], "Test Case 6: Errors should be empty when only numbers are present.")
        # Internal potential_numbers should be populated, potential_labels empty for "blood_pressure_monitor" specific keywords.

if __name__ == '__main__':
    unittest.main()
