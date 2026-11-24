"""
Fungsi-fungsi tahap pengolahan citra untuk pipeline usulan.
Setiap fungsi menerima dan mengembalikan citra dalam format numpy array (OpenCV, BGR atau grayscale).

TODO (Computer Vision Engineer): isi logika tiap fungsi.
"""
import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Ubah ke grayscale kalau citra masih berwarna, kalau sudah grayscale dibiarkan."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image
 
 
def _order_points(pts: np.ndarray) -> np.ndarray:
    """Urutkan 4 titik sudut: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    pts = pts.reshape(4, 2).astype("float32")
    jumlah = pts.sum(axis=1)             # x + y
    selisih = np.diff(pts, axis=1).ravel()  # y - x
 
    urut = np.zeros((4, 2), dtype="float32")
    urut[0] = pts[np.argmin(jumlah)]    # kiri-atas: x+y paling kecil
    urut[2] = pts[np.argmax(jumlah)]    # kanan-bawah: x+y paling besar
    urut[1] = pts[np.argmin(selisih)]   # kanan-atas: y-x paling kecil
    urut[3] = pts[np.argmax(selisih)]   # kiri-bawah: y-x paling besar
    return urut

def correct_perspective(image: np.ndarray, min_area_ratio: float = 0.2) -> np.ndarray:
    """
    Mendeteksi tepi/kontur kertas pada foto, lalu melakukan transformasi
    perspektif agar kertas tampak lurus dan rata (bird's-eye view).

    TODO: deteksi kontur terbesar -> approxPolyDP -> getPerspectiveTransform -> warpPerspective
    """
    
    gray = _to_gray(image)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
 
    # Cari tepi, lalu tebalkan supaya garis tepi kertas tersambung jadi satu kontur
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
 
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image
 
    luas_foto = image.shape[0] * image.shape[1]
 
    # Periksa beberapa kontur terbesar, ambil yang pertama berbentuk segi empat
    for kontur in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        if cv2.contourArea(kontur) < min_area_ratio * luas_foto:
            break
        keliling = cv2.arcLength(kontur, True)
        pendekatan = cv2.approxPolyDP(kontur, 0.02 * keliling, True)
 
        if len(pendekatan) == 4:
            kiri_atas, kanan_atas, kanan_bawah, kiri_bawah = _order_points(pendekatan)
 
            # Ukuran hasil = sisi terpanjang untuk lebar dan tinggi
            lebar = int(max(np.linalg.norm(kanan_bawah - kiri_bawah),
                            np.linalg.norm(kanan_atas - kiri_atas)))
            tinggi = int(max(np.linalg.norm(kanan_atas - kanan_bawah),
                             np.linalg.norm(kiri_atas - kiri_bawah)))
 
            asal = np.array([kiri_atas, kanan_atas, kanan_bawah, kiri_bawah], dtype="float32")
            tujuan = np.array([[0, 0], [lebar - 1, 0],
                               [lebar - 1, tinggi - 1], [0, tinggi - 1]], dtype="float32")
 
            matriks = cv2.getPerspectiveTransform(asal, tujuan)
            return cv2.warpPerspective(image, matriks, (lebar, tinggi))
 
    return image
    

    #raise NotImplementedError


def denoise_and_sharpen(image: np.ndarray, amount: float = 1.0, sigma: float = 2.0) -> np.ndarray:
    """
    Mengurangi noise (misalnya Gaussian blur ringan) lalu mempertajam
    tepi karakter (unsharp masking).

    TODO: GaussianBlur untuk denoise, lalu unsharp mask untuk sharpening
    """
    # 1. Denoise: blur kecil untuk meredam noise halus
    denoised = cv2.GaussianBlur(image, (3, 3), 0)
 
    # 2. Unsharp mask: hasil = asli*(1+amount) - blur*amount
    blurred = cv2.GaussianBlur(denoised, (0, 0), sigma)

    return cv2.addWeighted(denoised, 1.0 + amount, blurred, -amount, 0)
    
    #raise NotImplementedError


def adaptive_threshold(image: np.ndarray, block_size: int = 35, c: int = 15) -> np.ndarray:
    """
    Binarisasi adaptif (cv2.adaptiveThreshold) — lebih tahan terhadap
    pencahayaan tidak merata dibanding Otsu global.

    TODO: convert ke grayscale jika perlu, lalu cv2.adaptiveThreshold
    """
    gray = _to_gray(image)
 
    # block_size wajib ganjil dan lebih dari 1
    if block_size % 2 == 0:
        block_size += 1
    block_size = max(block_size, 3)
 
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size, c,
    )

    #raise NotImplementedError


def apply_morphology(image: np.ndarray, kernel_size: int = 2) -> np.ndarray:
    """
    Operasi morfologi (opening/closing) untuk membersihkan noise kecil
    hasil thresholding, tanpa merusak bentuk karakter.

    TODO: cv2.morphologyEx dengan kernel kecil (misal 2x2 atau 3x3)
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
 
    terbalik = cv2.bitwise_not(image)
    # Opening: hapus bintik putih kecil (noise) di luar tulisan
    bersih = cv2.morphologyEx(terbalik, cv2.MORPH_OPEN, kernel)
    # Closing: sambung celah kecil di dalam goresan huruf
    bersih = cv2.morphologyEx(bersih, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(bersih)

    #raise NotImplementedError
