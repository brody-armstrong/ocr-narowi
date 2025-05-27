import unittest
import tempfile
import os
import shutil
import cv2
import numpy as np
import logging
import io
import re
import subprocess
import sys

# Attempt to import the script to be tested
try:
    import narowi # Used for direct calls if any, and potentially type hints
except ImportError:
    print("Failed to import narowi. Make sure it's in the PYTHONPATH or same directory.")
    narowi = None # Set to None if import fails, script execution will be primary test method
    # raise # Optionally re-raise if direct module usage is critical

# Path to the narowi.py script, assuming it's in the parent directory or same directory
# Adjust if narowi.py is located elsewhere relative to the test script.
NAROWI_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "..", "narowi.py")
if not os.path.exists(NAROWI_SCRIPT_PATH):
    NAROWI_SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "narowi.py") # Try same directory


class TestOCRProcessing(unittest.TestCase):

    def _create_base_image(self, file_path, text_to_write, width=450, height=100, 
                           bg_color=(255, 255, 255), text_color=(0, 0, 0), border_color=None, border_thickness=1):
        """
        Base function to create an image with specified text, colors, and optional border.
        """
        image = np.full((height, width, 3), bg_color, dtype=np.uint8)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        thickness = 3
        
        text_size, _ = cv2.getTextSize(text_to_write, font, font_scale, thickness)
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        text_x = max(text_x, 10) 
        text_y = max(text_y, 10)

        if border_color is not None and border_thickness > 0:
            # Draw border by drawing slightly larger text in border_color first
            for dx in range(-border_thickness, border_thickness + 1):
                for dy in range(-border_thickness, border_thickness + 1):
                    if dx*dx + dy*dy <= border_thickness*border_thickness: # circular-ish border
                        cv2.putText(image, text_to_write, (text_x + dx, text_y + dy), font, font_scale, border_color, thickness)
        
        cv2.putText(image, text_to_write, (text_x, text_y), font, font_scale, text_color, thickness)
        cv2.imwrite(file_path, image)
        if not os.path.exists(file_path):
            raise IOError(f"Failed to write image to {file_path}")


    def create_dark_on_light_image(self, file_path, text_to_write, width=450, height=100):
        """Creates a standard dark text on light background image."""
        self._create_base_image(file_path, text_to_write, width, height, 
                                bg_color=(255, 255, 255), text_color=(0, 0, 0))

    def create_light_on_dark_image(self, file_path, text_to_write, width=450, height=100):
        """Creates a light text on dark background image."""
        self._create_base_image(file_path, text_to_write, width, height, 
                                bg_color=(30, 30, 30), text_color=(225, 225, 225))

    def create_problematic_contrast_image(self, file_path, text_to_write, width=450, height=100):
        """
        Creates an image with low contrast: light gray digit with a slightly lighter gray border 
        on a slightly darker light gray background.
        Digit: 180, Border: 200, Background: 170
        """
        self._create_base_image(file_path, text_to_write, width, height,
                                bg_color=(170, 170, 170),         # Light gray background
                                text_color=(180, 180, 180),       # Lighter gray text
                                border_color=(200, 200, 200),     # Even lighter gray border
                                border_thickness=1)


    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        # Logging setup is removed as we will capture stdout from script execution

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def _run_narowi_script_on_folder(self, folder_path):
        """Runs the narowi.py script on the given folder and returns its stdout and stderr."""
        if not narowi: # If import failed
             self.skipTest("narowi module not available for import, cannot determine script path reliably for subprocess.")

        if not os.path.exists(NAROWI_SCRIPT_PATH):
            self.fail(f"narowi.py script not found at {NAROWI_SCRIPT_PATH}. Please check the path.")

        try:
            process = subprocess.run(
                [sys.executable, NAROWI_SCRIPT_PATH, folder_path],
                capture_output=True, text=True, check=True, timeout=30 # Added timeout
            )
            return process.stdout, process.stderr
        except subprocess.CalledProcessError as e:
            self.fail(f"Script execution failed: {e}\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")
        except subprocess.TimeoutExpired as e:
            self.fail(f"Script execution timed out: {e}\nStdout:\n{e.stdout}\nStderr:\n{e.stderr}")


    def _parse_extracted_numbers_from_output(self, output):
        """Parses the 'Extracted Numbers' section from the script's stdout."""
        extracted_numbers = []
        in_extracted_section = False
        # Pattern to match: "Value: <val>, Raw Text: '<raw>', Confidence: <conf>%, BBox: [<bbox>]"
        # Making it more flexible for variations in float/int for value and confidence
        pattern = re.compile(r"Value: ([\d\.-]+),\s+Raw Text: '([^']*)',\s+Confidence: [\d\.]+%,\s+BBox: \[.+\]")

        for line in output.splitlines():
            line = line.strip()
            if line.startswith("Extracted Numbers:"):
                in_extracted_section = True
                continue
            if in_extracted_section:
                if not line or line.startswith("--") or line.startswith("Processing image:"): # End of section or new image
                    break 
                match = pattern.search(line)
                if match:
                    value_str, raw_text = match.groups()
                    try:
                        value = float(value_str) if '.' in value_str else int(value_str)
                    except ValueError:
                        self.fail(f"Could not parse value '{value_str}' from output: {line}")
                    extracted_numbers.append({'value': value, 'raw_text': raw_text})
        return extracted_numbers

    def test_dark_on_light_extraction(self):
        test_image_name = "test_dark_on_light.png"
        test_image_path = os.path.join(self.test_dir, test_image_name)
        expected_numbers = [{'value': 123, 'raw_text': "123"}, {'value': 789, 'raw_text': "789"}]
        
        self.create_dark_on_light_image(test_image_path, "123 789")
        
        stdout, stderr = self._run_narowi_script_on_folder(self.test_dir)
        self.assertNotIn("Error", stderr, f"Script stderr should be empty but was: {stderr}")
        
        extracted_data = self._parse_extracted_numbers_from_output(stdout)
        
        self.assertEqual(len(extracted_data), len(expected_numbers), 
                         f"Expected {len(expected_numbers)} numbers, got {len(extracted_data)}. Output:\n{stdout}")

        for expected_num in expected_numbers:
            self.assertIn(expected_num, extracted_data,
                          f"Expected number {expected_num} not found in extracted data {extracted_data}. Output:\n{stdout}")

    def test_light_on_dark_image_processing(self):
        test_image_name = "test_light_on_dark.png"
        test_image_path = os.path.join(self.test_dir, test_image_name)
        expected_number = {'value': 789, 'raw_text': "789"}
        
        self.create_light_on_dark_image(test_image_path, "789")
        
        stdout, stderr = self._run_narowi_script_on_folder(self.test_dir)
        self.assertNotIn("Error", stderr, f"Script stderr should be empty but was: {stderr}")
        
        extracted_data = self._parse_extracted_numbers_from_output(stdout)
        
        self.assertIn(expected_number, extracted_data,
                      f"Expected number {expected_number} not found in extracted data {extracted_data}. Output:\n{stdout}")

    def test_problematic_contrast_image_processing(self):
        test_image_name = "test_problematic_contrast.png"
        test_image_path = os.path.join(self.test_dir, test_image_name)
        # Note: OCR on such images can be very challenging. This test might be flaky.
        # The specific values for colors in create_problematic_contrast_image might need tuning
        # if this test consistently fails.
        expected_number = {'value': 456, 'raw_text': "456"} 
        
        self.create_problematic_contrast_image(test_image_path, "456")
        
        stdout, stderr = self._run_narowi_script_on_folder(self.test_dir)
        # It's possible that problematic contrast leads to no numbers found, or errors in script.
        # For now, let's check stderr isn't showing critical script failures.
        # The main check is whether the number is found.
        # self.assertNotIn("Error: could not read image", stderr) # Example specific check
        
        extracted_data = self._parse_extracted_numbers_from_output(stdout)
        
        # For problematic images, we might not always get the number.
        # The goal is to see if our preprocessing helps. If it's consistently missed,
        # it indicates a need for preprocessing tuning or that the case is too hard.
        # For now, we assert it IS found. If it's flaky, we might reconsider the assertion.
        self.assertIn(expected_number, extracted_data,
                      f"Expected number {expected_number} not found in extracted data for problematic contrast image. Output:\n{stdout}\nStderr:\n{stderr}")


if __name__ == '__main__':
    unittest.main()
