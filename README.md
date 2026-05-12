# Kalkulator Inbreeding Sapi - Final Fix

Versi ini dibuat untuk memperbaiki kegagalan perhitungan dan masalah nilai kosong/NaN pada hasil.

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Format data

```csv
Animal_ID,Sire_ID,Dam_ID
I,-,-
P,-,-
C,I,P
D,I,P
X,I,P
B,D,C
A,B,C
E,B,-
F,D,-
```

## Hasil contoh dari gambar

- B = D x C -> F = 25%
- A = B x C -> F = 37,5%

## Catatan

Gunakan tanda `-` untuk parent yang tidak diketahui.
