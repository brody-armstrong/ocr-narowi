import cv2
import numpy as np
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
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error loading image: File not found or corrupted at {image_path}")
                return None
            return img
        except Exception as e:
            print(f"Error loading image at {image_path}: {e}")
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
    def preprocess_for_ocr(image: np.ndarray, 
                           gaussian_kernel_size: Tuple[int, int] = (5, 5),
                           use_otsu: bool = False,
                           morph_kernel_size: Optional[Tuple[int, int]] = None
                           ) -> np.ndarray:
        """
        Apply a preprocessing pipeline for OCR.

        Args:
            image: Input image (expected to be BGR if it has color).
            gaussian_kernel_size: Kernel size for Gaussian blur. Default is (5, 5).
            use_otsu: If True, use Otsu's thresholding instead of adaptive thresholding.
                      Default is False.
            morph_kernel_size: Kernel size for morphological opening (erosion then dilation).
                               If None, this step is skipped. Example: (3,3). Default is None.
                               
        Returns:
            numpy.ndarray: Preprocessed image (binary).
        """
        print("Starting OCR preprocessing...")
        # Convert to grayscale
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print("Converted image to grayscale.")
        elif len(image.shape) == 2:
            gray = image # Already grayscale
            print("Image is already grayscale.")
        else: # Fallback for unexpected shapes, or could raise an error
            gray = image 
            print(f"Warning: Image has unexpected shape {image.shape}. Attempting to process as is.")

        # Apply Gaussian blur to reduce noise
        print(f"Applying Gaussian blur with kernel: {gaussian_kernel_size}")
        blurred = cv2.GaussianBlur(gray, gaussian_kernel_size, 0)
        
        # Apply thresholding
        if use_otsu:
            print("Applying Otsu's thresholding...")
            # Otsu's thresholding automatically finds the optimal threshold value
            _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            print("Applying adaptive Gaussian thresholding...")
            thresh = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2  # Default parameters for adaptive threshold
            )
        
        # Apply morphological operations if kernel size is provided
        if morph_kernel_size:
            # Ensure morph_kernel_size has positive integer values
            if not (isinstance(morph_kernel_size, tuple) and 
                    len(morph_kernel_size) == 2 and
                    isinstance(morph_kernel_size[0], int) and morph_kernel_size[0] > 0 and
                    isinstance(morph_kernel_size[1], int) and morph_kernel_size[1] > 0):
                print(f"Invalid morph_kernel_size: {morph_kernel_size}. Skipping morphological operations.")
            else:
                print(f"Applying morphological opening with kernel: {morph_kernel_size}")
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
                # Erosion followed by Dilation (Opening)
                eroded = cv2.erode(thresh, kernel, iterations=1)
                print("Applied erosion.")
                dilated = cv2.dilate(eroded, kernel, iterations=1)
                print("Applied dilation.")
                thresh = dilated # Update thresh with the result of morphological operations
        else:
            print("Skipping morphological operations.")
            
        print("OCR preprocessing finished.")
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
            print(f"Image saved successfully to {output_path}")
            return True
        except Exception as e:
            print(f"Error saving image to {output_path}: {e}")
            return False
