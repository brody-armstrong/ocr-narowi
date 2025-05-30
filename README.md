# Narowi

Narowi is a medical device data extraction system that uses OCR (Optical Character Recognition) to extract and categorize medical readings from images. The project is divided into phases, with each phase focusing on a specific aspect of the system.

## Project Overview

- **Phase 1**: Implemented OCR using Python and Tesseract OCR, including modules for image processing, OCR engine wrapping, and number extraction.
- **Phase 2**: Focuses on ROI (Region of Interest) detection and medical pattern recognition. This phase includes:
  - ROI detector module for identifying regions of interest in images.
  - Pattern matcher module for categorizing medical readings (e.g., blood pressure, temperature, weight, oxygen, heart rate) from extracted text.

## Current State

The current codebase includes:

- **Image Processing**: Modules for preprocessing images to improve OCR accuracy.
- **OCR Engine**: A wrapper around Tesseract OCR for text extraction.
- **ROI Detector**: A module for detecting regions of interest in images.
- **Pattern Matcher**: A module for identifying and categorizing medical readings from extracted text.

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Tesseract OCR installed on your system

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/narowi.git
   cd narowi
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the tests to ensure everything is working:
   ```bash
   python -m pytest tests/
   ```

## Usage

To use the OCR system, you can run the main script:

```bash
python medical_ocr/src/main.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 