# Kalkulator Inbreeding Sapi - Streamlit

Aplikasi ini menghitung koefisien inbreeding sapi dari data pedigree.

## Perbaikan versi ini

- Tampilan tidak lagi menampilkan `NaN`.
- Parent tidak diketahui ditampilkan sebagai `-`.
- Pembacaan CSV/Excel memakai `keep_default_na=False`, sehingga sel kosong tidak otomatis berubah menjadi NaN.
- Contoh data sapi sudah memiliki nilai inbreeding yang jelas.
- Hasil ditampilkan sebagai tabel, proses perhitungan, grafik, dan bagan pedigree.

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Contoh data

```csv
Animal_ID,Sire_ID,Dam_ID
SAPI_JANTAN_01,-,-
SAPI_BETINA_01,-,-
SAPI_A1,SAPI_JANTAN_01,SAPI_BETINA_01
SAPI_A2,SAPI_JANTAN_01,SAPI_BETINA_01
SAPI_B1,SAPI_A1,SAPI_A2
```

SAPI_B1 adalah hasil perkawinan saudara kandung penuh, sehingga F = 25%.
