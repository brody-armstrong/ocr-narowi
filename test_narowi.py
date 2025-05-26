import unittest
import tempfile
import os
import shutil
import cv2
import numpy as np
import logging
import io

# Attempt to import the script to be tested
try:
    import narowi
except ImportError:
    print("Failed to import narowi. Make sure it's in the PYTHONPATH or same directory.")
    raise

class TestOCRProcessing(unittest.TestCase):

    def create_test_image(self, file_path, text_to_write, width=450, height=100): # Wider for "123 789"
        """
        Creates an image with specified text, e.g., "123 789".
        """
        image = np.ones((height, width, 3), dtype=np.uint8) * 255 # White background

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2  # Scale for "123 789"
        font_color = (0, 0, 0)  # Black
        thickness = 3    # Thickness
        
        text_size = cv2.getTextSize(text_to_write, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        text_x = max(text_x, 10) 
        text_y = max(text_y, 10)

        cv2.putText(image, text_to_write, (text_x, text_y), font, font_scale, font_color, thickness)
        cv2.imwrite(file_path, image)

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.test_image_name = "test_ocr_image.png"
        self.test_image_path = os.path.join(self.test_dir, self.test_image_name)
        
        self.expected_text_in_image = "123 789" # Original suggested text
        
        self.create_test_image(self.test_image_path, self.expected_text_in_image)

        self.logger = logging.getLogger() 
        self.original_handlers = list(self.logger.handlers)
        self.original_level = self.logger.level
        
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        self.logger.setLevel(logging.INFO) 
        
        self.log_stream = io.StringIO()
        self.stream_handler = logging.StreamHandler(self.log_stream)
        self.stream_handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s:%(message)s'))
        self.logger.addHandler(self.stream_handler)

    def tearDown(self):
        self.logger.removeHandler(self.stream_handler)
        self.stream_handler.close()
        
        for handler in self.logger.handlers[:]: 
            self.logger.removeHandler(handler)
        for handler in self.original_handlers: 
            self.logger.addHandler(handler)
        
        self.logger.setLevel(self.original_level)
        
        shutil.rmtree(self.test_dir)

    def test_process_single_image_correct_text(self):
        narowi.process_images_from_folder(self.test_dir)
        log_contents = self.log_stream.getvalue()
        
        detected_text_found = None
        
        for line in log_contents.splitlines():
            log_message_prefix = f"INFO:root:Detected text in {self.test_image_name}:"
            if line.startswith(log_message_prefix):
                detected_text_segment = line.split(log_message_prefix, 1)[1].strip()
                detected_text_found = detected_text_segment
                break
        
        self.assertIsNotNone(detected_text_found, 
                             f"Log message for detected text in '{self.test_image_name}' not found. Log content:\n{log_contents}")
        
        # Normalize by splitting and rejoining to handle variable spacing from OCR
        normalized_detected_text = " ".join(detected_text_found.split())
        normalized_expected_text = " ".join(self.expected_text_in_image.split())

        self.assertEqual(normalized_detected_text, normalized_expected_text,
                         f"Detected text '{normalized_detected_text}' does not match expected '{normalized_expected_text}'. Log content:\n{log_contents}")

if __name__ == '__main__':
    unittest.main()
