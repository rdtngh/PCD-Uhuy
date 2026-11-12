# Tugas Besar Pengolahan Citra Digital (IF25-40410)

## Judul
Perbaikan Kualitas Citra Foto Catatan Kuliah untuk Peningkatan Akurasi Optical Character Recognition Berbasis Koreksi Perspektif dan Adaptive Thresholding

## Ringkasan
Pipeline pengolahan citra untuk memperbaiki foto catatan kuliah hasil kamera ponsel (normal, miring, pencahayaan redup, bayangan sebagian, blur ringan), sehingga akurasi OCR (Tesseract) meningkat dibanding citra mentah / baseline (Otsu).

Tahap pipeline: koreksi perspektif → denoise/sharpening → adaptive thresholding → morfologi → OCR.

## Anggota Kelompok

| Nama | NIM | Peran |
| --- | --- | --- |
| Raditya Alrasyid Nugroho | 123140125 | Project Lead |
| Bening Apni Prameswari | 123140089 | Data Engineer & Annotator |
| Ahmad Aufamahdi Salam | 123140092 | Computer Vision Engineer |
| Muhammad Fadhel | 123140106 | Evaluation & QA Analyst |
| Rian Rafael Sangap Tamba | 123140192 | Application Engineer |

## Struktur Repo

```
PCD-Uhuy/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/               # foto asli (tidak di-commit penuh, lihat catatan di bawah)
│   ├── ground_truth/      # teks ketikan ulang (.txt)
│   └── metadata.csv       # label kondisi tiap foto
├── src/
│   ├── preprocessing.py   # perspektif, sharpening, threshold, morfologi
│   ├── baseline.py        # pipeline baseline (Otsu + OCR)
│   ├── pipeline.py        # pipeline usulan, tiap tahap bisa dinyalakan/dimatikan
│   ├── ocr.py             # pemanggilan Tesseract
│   └── evaluate.py        # hitung CER/WER, jalankan baseline & ablasi
├── notebooks/             # eksplorasi, tuning parameter, hasil ablasi
├── results/               # tabel CSV dan grafik hasil evaluasi
└── app/
    └── app.py             # demo Streamlit
```

## Instalasi

```bash
pip install -r requirements.txt
```

Tesseract OCR harus diinstal terpisah (bukan lewat pip):

- **Ubuntu/Colab:** `apt install tesseract-ocr`
- **Windows:** unduh installer dari https://github.com/UB-Mannheim/tesseract/wiki
- **macOS:** `brew install tesseract`

## Dataset

Foto asli (data primer) disimpan di Google Drive bersama kelompok karena ukurannya besar: [tautan Drive Menyusul]

`data/metadata.csv` berisi label kondisi tiap foto: `normal`, `miring`, `pencahayaan_redup`, `bayangan_sebagian`, `blur_ringan`.

## Menjalankan Demo

```bash
streamlit run app/app.py
```

## Evaluasi dan Ablasi

```bash
python src/evaluate.py
```

Menghasilkan tabel CER/WER untuk baseline, pipeline lengkap, dan setiap skenario ablasi di `results/`.
