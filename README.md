# Software Inbreeding Sapi - Fix Total

Versi ini memperbaiki masalah hasil yang masih tampil kosong/NaN dan menambahkan perhitungan kondisi inbreeding.

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Fitur

- Perhitungan otomatis setelah data masuk
- Tidak menampilkan nilai kosong sebagai NaN
- Parent tidak diketahui ditampilkan sebagai tanda `-`
- Kolom hasil:
  - Hubungan_Parent_A
  - Koefisien_Inbreeding_F
  - Inbreeding_%
  - Kondisi_Inbreeding
  - Rekomendasi
  - Proses_Perhitungan
- Grafik nilai inbreeding
- Bagan pedigree
- Contoh dari gambar langsung tersedia di aplikasi

## Contoh dari gambar

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

Hasil utama:
- B = D x C -> F = 25%
- A = B x C -> F = 37,5%
