"""
Pipeline baseline: grayscale + Otsu threshold, tanpa koreksi perspektif,
tanpa sharpening, tanpa morfologi. Dipakai sebagai pembanding utama.

TODO (Computer Vision Engineer): isi logika.
"""
import cv2
import numpy as np


def baseline_pipeline(image: np.ndarray) -> np.ndarray:
    """
    TODO: grayscale -> cv2.threshold dengan cv2.THRESH_OTSU
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
 
    # Angka 0 di sini diabaikan karena ada flag THRESH_OTSU,
    # batas yang dipakai dihitung otomatis oleh Otsu.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
    # raise NotImplementedError
