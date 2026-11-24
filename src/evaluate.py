"""
Perhitungan CER/WER dan skenario ablasi.

TODO (Evaluation & QA Analyst): isi logika, baca data dari data/metadata.csv
dan data/ground_truth/, jalankan baseline + pipeline usulan + 4 skenario ablasi,
simpan hasil ke results/.
"""
import argparse
import os
 
import cv2
import jiwer
import numpy as np
import pandas as pd
import pytesseract
 
from .baseline import baseline_pipeline
from .ocr import run_ocr
from .pipeline import proposed_pipeline
 
# Lokasi data (sesuai struktur folder proyek)
METADATA_PATH = "data/metadata 100 citra.csv"
IMAGES_DIR = "data/images"
GROUND_TRUTH_DIR = "data/ground truth 100 citra"
RESULTS_DIR = "results"

SKENARIO = {
    "Baseline (grayscale + Otsu)": baseline_pipeline,
    "Pipeline lengkap": proposed_pipeline,
    "Tanpa perspektif": lambda img: proposed_pipeline(img, use_perspective=False),
    "Tanpa sharpening": lambda img: proposed_pipeline(img, use_sharpening=False),
    "Adaptive diganti Otsu": lambda img: proposed_pipeline(img, use_adaptive=False),
    "Tanpa morfologi": lambda img: proposed_pipeline(img, use_morphology=False),
}

 
def _normalisasi(teks: str) -> str:
    """Rapikan spasi dan baris baru jadi satu spasi, supaya selisih format tidak dihitung salah."""
    return " ".join(teks.split())


def _cek_bahasa(lang: str) -> None:
    """Pastikan data bahasa Tesseract terinstal, supaya error-nya jelas dan muncul di awal."""
    tersedia = set(pytesseract.get_languages(config=""))
    kurang = set(lang.split("+")) - tersedia
    if kurang:
        raise RuntimeError(
            f"Data bahasa Tesseract belum terinstal: {', '.join(sorted(kurang))}.\n"
            "  Arch   : sudo pacman -S tesseract-data-ind\n"
            "  Ubuntu : sudo apt install tesseract-ocr-ind\n"
            "  macOS  : brew install tesseract-lang\n"
            "Atau jalankan dengan bahasa lain, misalnya: --lang eng"
        )

def compute_cer_wer(reference: str, hypothesis: str) -> dict:
    """
    Menghitung CER dan WER antara teks ground truth (reference) dan
    hasil OCR (hypothesis) menggunakan library jiwer.

    TODO: jiwer.cer(reference, hypothesis), jiwer.wer(reference, hypothesis)
    """
    ref = _normalisasi(reference)
    hyp = _normalisasi(hypothesis)
 
    if not ref:
        raise ValueError("Teks ground truth kosong, CER/WER tidak bisa dihitung.")
    if not hyp:
        return {"cer": 1.0, "wer": 1.0}
 
    return {"cer": jiwer.cer(ref, hyp), "wer": jiwer.wer(ref, hyp)}
    
    # raise NotImplementedError

def run_all_scenarios(
    metadata_path: str = METADATA_PATH,
    images_dir: str = IMAGES_DIR,
    gt_dir: str = GROUND_TRUTH_DIR,
    lang: str = "ind",
    split: str = None,
    limit: int = None,
) -> pd.DataFrame:
    """
    Menjalankan baseline, pipeline lengkap, dan skenario ablasi
    (tanpa perspektif, tanpa sharpening, adaptive->Otsu, tanpa morfologi)
    pada seluruh data uji, lalu mengembalikan tabel hasil CER/WER per skenario.

    TODO: loop semua foto di metadata, jalankan tiap skenario, kumpulkan hasil.
    """

    _cek_bahasa(lang)

    meta = pd.read_csv(metadata_path)
    if split is not None:
        meta = meta[meta["split"] == split]
    if limit is not None:
        meta = meta.head(limit)
 
    total = len(meta)
    cache_gt = {}   # 5 foto dari catatan yang sama memakai satu file ground truth
    baris = []
 
    for i, row in enumerate(meta.itertuples(index=False), start=1):
        print(f"[{i}/{total}] {row.filename}", flush=True)
 
        foto = cv2.imread(os.path.join(images_dir, row.filename))
        if foto is None:
            raise FileNotFoundError(f"Foto tidak ditemukan/tidak terbaca: {row.filename}")
 
        if row.ground_truth not in cache_gt:
            path_gt = os.path.join(gt_dir, row.ground_truth)
            with open(path_gt, encoding="utf-8") as f:
                teks_gt = f.read()
            if not _normalisasi(teks_gt):
                raise ValueError(f"File ground truth kosong: {path_gt}")
            cache_gt[row.ground_truth] = teks_gt
        referensi = cache_gt[row.ground_truth]
 
        for nama, fungsi in SKENARIO.items():
            try:
                citra = fungsi(foto)
                teks = _normalisasi(run_ocr(citra, lang=lang))
                skor = compute_cer_wer(referensi, teks)
            except Exception as err:  # satu skenario gagal tidak boleh menghentikan 600 proses
                print(f"   ! gagal di '{nama}': {err}", flush=True)
                teks = ""
                skor = {"cer": np.nan, "wer": np.nan}
 
            baris.append({
                "filename": row.filename,
                "page_id": row.page_id,
                "condition": row.condition,
                "split": row.split,
                "scenario": nama,
                "cer": skor["cer"],
                "wer": skor["wer"],
                "ocr_text": teks,
            })
 
    return pd.DataFrame(baris)

    # raise NotImplementedError



def ringkas_per_skenario(df: pd.DataFrame) -> pd.DataFrame:
    """Rata-rata CER/WER per skenario (inilah tabel ablasi utama)."""
    return (
        df.groupby("scenario", sort=False)
        .agg(cer_mean=("cer", "mean"), wer_mean=("wer", "mean"), n_foto=("cer", "count"))
        .round(4)
        .reset_index()
    )
 
 
def ringkas_per_kondisi(df: pd.DataFrame, metrik: str = "cer") -> pd.DataFrame:
    """Rata-rata CER (atau WER) per kondisi foto (baris) per skenario (kolom)."""
    tabel = df.pivot_table(index="condition", columns="scenario", values=metrik, aggfunc="mean")
    urutan_kolom = [s for s in SKENARIO if s in tabel.columns]
    urutan_baris = [k for k in df["condition"].unique() if k in tabel.index]
    return tabel.reindex(index=urutan_baris, columns=urutan_kolom).round(4)
 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Evaluasi CER/WER baseline, pipeline, dan ablasi")
    parser.add_argument("--lang", default="ind", help="bahasa Tesseract (default: ind)")
    parser.add_argument("--split", default=None, help="pakai hanya split ini, misal: development")
    parser.add_argument("--limit", type=int, default=None, help="pakai N foto pertama saja")
    args = parser.parse_args()
 
    df = run_all_scenarios(lang=args.lang, split=args.split, limit=args.limit)
 
    ringkasan = ringkas_per_skenario(df)
    cer_kondisi = ringkas_per_kondisi(df, "cer")
    wer_kondisi = ringkas_per_kondisi(df, "wer")
 
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(f"{RESULTS_DIR}/detail_per_foto.csv", index=False)
    ringkasan.to_csv(f"{RESULTS_DIR}/ablation_results.csv", index=False)
    cer_kondisi.to_csv(f"{RESULTS_DIR}/per_kondisi_cer.csv")
    wer_kondisi.to_csv(f"{RESULTS_DIR}/per_kondisi_wer.csv")
 
    print("\n=== Rata-rata per skenario ===")
    print(ringkasan.to_string(index=False))
    print("\n=== CER per kondisi ===")
    print(cer_kondisi.to_string())
