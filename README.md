# Aplikasi Streamlit Kalkulator Inbreeding Pedigree Ternak

Aplikasi ini menghitung koefisien inbreeding dari data pedigree ternak.

## Fitur

- Upload data CSV atau Excel `.xlsx`
- Data kosong tidak lagi tampil sebagai `NaN`, tetapi sebagai `-`
- Menghitung:
  - Koefisien inbreeding `F`
  - Persentase inbreeding
  - Hubungan parent `A(sire, dam)`
  - Proses perhitungan per ternak
- Menampilkan:
  - Tabel hasil
  - Ringkasan hasil
  - Bagan/pohon pedigree
  - Grafik ternak dengan nilai inbreeding tertinggi
- Download hasil CSV dan Excel

## Cara menjalankan

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Format Data

Minimal harus ada 3 kolom:

| Kolom | Keterangan |
|---|---|
| Animal_ID | ID ternak |
| Sire_ID | ID pejantan/bapak |
| Dam_ID | ID betina/induk |

Contoh:

```csv
Animal_ID,Sire_ID,Dam_ID
P1,,
P2,,
A1,P1,P2
A2,P1,P2
B1,A1,A2
```

## Contoh Proses Inbreeding

P1 dan P2 adalah founder.

A1 dan A2 sama-sama anak dari P1 x P2, sehingga A1 dan A2 adalah saudara kandung penuh.

Jika B1 adalah anak dari A1 x A2, maka:

```text
Hubungan A1 dan A2 = 0,50
F_B1 = 0,5 x 0,50 = 0,25 = 25%
```
