"""
Perhitungan CER/WER dan skenario ablasi.

TODO (Evaluation & QA Analyst): isi logika, baca data dari data/metadata.csv
dan data/ground_truth/, jalankan baseline + pipeline usulan + 4 skenario ablasi,
simpan hasil ke results/.
"""
import jiwer
import pandas as pd


def compute_cer_wer(reference: str, hypothesis: str) -> dict:
    """
    Menghitung CER dan WER antara teks ground truth (reference) dan
    hasil OCR (hypothesis) menggunakan library jiwer.

    TODO: jiwer.cer(reference, hypothesis), jiwer.wer(reference, hypothesis)
    """
    raise NotImplementedError


def run_all_scenarios(metadata_path: str = "data/metadata.csv") -> pd.DataFrame:
    """
    Menjalankan baseline, pipeline lengkap, dan skenario ablasi
    (tanpa perspektif, tanpa sharpening, adaptive->Otsu, tanpa morfologi)
    pada seluruh data uji, lalu mengembalikan tabel hasil CER/WER per skenario.

    TODO: loop semua foto di metadata, jalankan tiap skenario, kumpulkan hasil.
    """
    raise NotImplementedError


if __name__ == "__main__":
    df = run_all_scenarios()
    df.to_csv("results/ablation_results.csv", index=False)
    print(df)
