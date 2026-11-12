"""
Demo Streamlit: upload foto catatan, tampilkan hasil tiap tahap pipeline,
tampilkan teks OCR, bandingkan dengan baseline.

TODO (Application Engineer): isi logika UI dan pemanggilan pipeline.
Jalankan dengan: streamlit run app/app.py
"""
import streamlit as st

st.set_page_config(page_title="ClearNote PCD", page_icon="📝")

st.title("ClearNote — Perbaikan Citra Catatan Kuliah untuk OCR")
st.write("Unggah foto catatan kuliah untuk melihat hasil perbaikan citra dan OCR.")

uploaded_file = st.file_uploader("Upload foto catatan", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.info("TODO: proses foto dengan src.pipeline.proposed_pipeline dan src.baseline.baseline_pipeline, lalu tampilkan hasil tiap tahap + teks OCR dari keduanya untuk dibandingkan.")
