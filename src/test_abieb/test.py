import cv2
from baseline import baseline_pipeline
from ocr import run_ocr

foto = cv2.imread("image.png")
hasil = baseline_pipeline(foto)
cv2.imwrite("hasil_baseline.png", hasil)   # lihat hasil hitam-putihnya
print(run_ocr(hasil, lang="ind"))
