"""
Wrapper pemanggilan Tesseract OCR.

TODO (Computer Vision Engineer / Evaluation & QA): isi logika.
"""
import pytesseract
import numpy as np


def run_ocr(image: np.ndarray, lang: str = "ind") -> str:
    """
    Menjalankan Tesseract OCR pada citra yang sudah diproses.

    TODO: pytesseract.image_to_string(image, lang=lang)
    Catatan: pastikan paket bahasa Tesseract untuk Indonesia (ind) terinstal,
    atau gunakan 'eng' kalau catatan dalam bahasa Inggris / campuran.
    """
    raise NotImplementedError
