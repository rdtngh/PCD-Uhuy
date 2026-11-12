"""
Pipeline usulan dengan saklar per tahap, supaya ablasi tinggal mengganti argumen.

TODO (Computer Vision Engineer): panggil fungsi dari preprocessing.py sesuai saklar di bawah.
"""
import numpy as np
from . import preprocessing as pp


def proposed_pipeline(
    image: np.ndarray,
    use_perspective: bool = True,
    use_sharpening: bool = True,
    use_adaptive: bool = True,
    use_morphology: bool = True,
) -> np.ndarray:
    """
    Menjalankan pipeline usulan. Setiap tahap bisa dimatikan untuk skenario ablasi:

    - use_perspective=False  -> lewati koreksi perspektif
    - use_sharpening=False   -> lewati denoise/sharpening
    - use_adaptive=False     -> ganti adaptive threshold dengan Otsu (lihat baseline.py)
    - use_morphology=False   -> lewati operasi morfologi

    TODO: panggil pp.correct_perspective, pp.denoise_and_sharpen,
    pp.adaptive_threshold, pp.apply_morphology sesuai flag di atas.
    """
    raise NotImplementedError
