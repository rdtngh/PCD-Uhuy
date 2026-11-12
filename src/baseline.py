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
    raise NotImplementedError
