from setuptools import setup, find_packages

setup(
    name="medical_ocr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "opencv-python",
        "pytesseract",
    ],
    python_requires=">=3.8",
) 