# 🐂 Kalkulator Inbreeding Ternak (Livestock Inbreeding Calculator)

Aplikasi berbasis web untuk menghitung koefisien inbreeding ($F$) dan koefisien kekerabatan ($R$) pada ternak menggunakan Metode Matriks Hubungan Adiditf (Additive Relationship Matrix).

## ✨ Fitur Utama
- **Perhitungan Akurat**: Menggunakan algoritma matriks untuk memastikan nilai inbreeding tepat (misalnya: perkawinan saudara kandung menghasilkan $F = 25\%$).
- **Visualisasi Pedigree**: Menampilkan diagram silsilah (pedigree tree) secara interaktif.
- **Analisis Dampak**: Memberikan penjelasan mengenai risiko inbreeding terhadap performa ternak (Inbreeding Depression).
- **Ekspor Data**: Hasil perhitungan dapat diunduh dalam format CSV dan TXT.
- **Antarmuka Modern**: Dilengkapi dengan mode terang/gelap dan desain responsif.

## 🚀 Cara Menjalankan

1. **Clone repositori**:
   ```bash
   git clone https://github.com/adiorany3/inbreed.git
   cd inbreed
   ```

2. **Install dependensi**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**:
   ```bash
   streamlit run app.py
   ```

## 📊 Format Input Data (CSV)

Pastikan file CSV Anda memiliki kolom: `Animal_ID`, `Sire_ID` (Bapak), dan `Dam_ID` (Induk). Gunakan tanda `-` untuk tetua yang tidak diketahui.

Contoh isi file:
```csv
Animal_ID,Sire_ID,Dam_ID
ID01,-,-
ID02,-,-
ID03,ID01,ID02
ID04,ID01,ID02
ID05,ID03,ID04
```

## 🛡️ Lisensi

Distributed under the MIT License. See `LICENSE` for more information.

## 👨‍💻 Kontribusi

Kontribusi sangat terbuka! Silakan fork repositori ini dan buat pull request.

---
Developed with ❤️ by **Galuh Adi Insani**
