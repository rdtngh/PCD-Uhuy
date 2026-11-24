# cek 1: bandingin hasil OCR dengan teks asli
import pandas as pd
import cv2
from src import preprocessing as pp
from src.baseline import baseline_pipeline

d = pd.read_csv("results/detail_per_foto.csv")
print(d[d.filename == "foto 1.1.jpg"][["scenario", "cer", "ocr_text"]].to_string())

# cek 2: perspektif beneran jalan nggak, dan seberapa gede fotonya
print("test 2 \n\n")
f = cv2.imread("data/images/foto 1.2.jpg")
print("ukuran foto:", f.shape)
print("perspektif ngubah foto?", pp.correct_perspective(f).shape != f.shape)
cv2.imwrite("cek_baseline.png", baseline_pipeline(f))
cv2.imwrite("cek_adaptive.png", pp.adaptive_threshold(pp.denoise_and_sharpen(f)))