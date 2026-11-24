"""
Pipeline usulan dengan saklar per tahap, supaya ablasi tinggal mengganti argumen.

TODO (Computer Vision Engineer): panggil fungsi dari preprocessing.py sesuai saklar di bawah.
"""
import numpy as np
from . import preprocessing as pp
from .baseline import baseline_pipeline



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
    hasil = image
 
    # 1. Koreksi perspektif (masih berwarna)
    if use_perspective:
        hasil = pp.correct_perspective(hasil)
 
    # 2. Denoise + sharpening (masih berwarna)
    if use_sharpening:
        hasil = pp.denoise_and_sharpen(hasil)
 
    # 3. Thresholding: adaptive (usulan) atau Otsu (pembanding untuk ablasi)
    if use_adaptive:
        hasil = pp.adaptive_threshold(hasil)
    else:
        hasil = baseline_pipeline(hasil)
 
    # 4. Morfologi (butuh citra biner, jadi selalu setelah thresholding)
    if use_morphology:
        hasil = pp.apply_morphology(hasil)
 
    return hasil

    # raise NotImplementedError
