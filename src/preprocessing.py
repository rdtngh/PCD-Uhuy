"""
Fungsi-fungsi tahap pengolahan citra untuk pipeline usulan.
Setiap fungsi menerima dan mengembalikan citra dalam format numpy array (OpenCV, BGR atau grayscale).

TODO (Computer Vision Engineer): isi logika tiap fungsi.
"""
import cv2
import numpy as np


def correct_perspective(image: np.ndarray) -> np.ndarray:
    """
    Mendeteksi tepi/kontur kertas pada foto, lalu melakukan transformasi
    perspektif agar kertas tampak lurus dan rata (bird's-eye view).

    TODO: deteksi kontur terbesar -> approxPolyDP -> getPerspectiveTransform -> warpPerspective
    """
    raise NotImplementedError


def denoise_and_sharpen(image: np.ndarray) -> np.ndarray:
    """
    Mengurangi noise (misalnya Gaussian blur ringan) lalu mempertajam
    tepi karakter (unsharp masking).

    TODO: GaussianBlur untuk denoise, lalu unsharp mask untuk sharpening
    """
    raise NotImplementedError


def adaptive_threshold(image: np.ndarray, block_size: int = 35, c: int = 15) -> np.ndarray:
    """
    Binarisasi adaptif (cv2.adaptiveThreshold) — lebih tahan terhadap
    pencahayaan tidak merata dibanding Otsu global.

    TODO: convert ke grayscale jika perlu, lalu cv2.adaptiveThreshold
    """
    raise NotImplementedError


def apply_morphology(image: np.ndarray) -> np.ndarray:
    """
    Operasi morfologi (opening/closing) untuk membersihkan noise kecil
    hasil thresholding, tanpa merusak bentuk karakter.

    TODO: cv2.morphologyEx dengan kernel kecil (misal 2x2 atau 3x3)
    """
    raise NotImplementedError
