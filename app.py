import io
import tempfile
import pathlib
import graphviz
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image


# ============================================================
# FINAL FIX - KALKULATOR INBREEDING TERNAK
# ============================================================

EMPTY = "-"
UNKNOWN_VALUES = {
    "", " ", "-", "--", "0", "na", "n/a", "nan", "none", "null",
    "unknown", "tidak diketahui", "tidak ada", "kosong",
}


def is_unknown(value) -> bool:
    if value is None: return True
    try:
        if pd.isna(value): return True
    except: pass
    text = str(value).strip()
    return text.lower() in UNKNOWN_VALUES


def clean_id(value) -> Optional[str]:
    if is_unknown(value): return None
    text = str(value).strip()
    # Handle numeric IDs ending in .0 from Excel
    if text.endswith(".0"):
        try:
            num = float(text)
            if num.is_integer(): text = str(int(num))
        except: pass
    return text


def show_value(value) -> str:
    if is_unknown(value): return EMPTY
    return str(value).strip()


def clean_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        else:
            out[col] = out[col].astype(str).apply(lambda x: EMPTY if is_unknown(x) else x)
    return out


def kondisi_inbreeding(percent: float) -> str:
    if percent <= 0: return "Tidak inbred"
    if percent < 6.25: return "Inbreeding rendah"
    if percent < 12.5: return "Inbreeding sedang"
    if percent < 25: return "Inbreeding tinggi"
    return "Inbreeding sangat tinggi"


def rekomendasi(percent: float) -> str:
    if percent <= 0: return "Aman berdasarkan pedigree: tidak terdeteksi inbreeding."
    if percent < 6.25: return "Masih rendah, tetapi tetap perlu monitoring."
    if percent < 12.5: return "Perlu perhatian. Hindari pengulangan perkawinan kerabat."
    if percent < 25: return "Risiko tinggi. Gunakan pasangan dari garis keturunan berbeda."
    return "Risiko sangat tinggi. Perkawinan kerabat dekat sebaiknya dihindari."


def dampak_inbreeding(percent: float) -> str:
    """Memberikan informasi tambahan mengenai dampak biologis berdasarkan persentase F."""
    if percent <= 0:
        return "**Dampak:** Variabilitas genetik terjaga maksimal. Tidak ada risiko depresi inbreeding terdeteksi."
    elif percent < 6.25:
        return "**Dampak:** Pengaruh minimal pada performa. Efek kumulatif mulai muncul jika terjadi terus-menerus dalam beberapa generasi."
    elif percent < 12.5:
        return "**Dampak:** Potensi penurunan performa produksi (pertumbuhan, produksi susu) sekitar 2-5%. Sedikit peningkatan risiko penyakit genetik resesif."
    elif percent < 25:
        return "**Dampak:** Depresi inbreeding mulai terlihat nyata. Penurunan fertilitas, daya tahan tubuh melemah, dan potensi munculnya cacat lahir (lethal traits)."
    else:
        return "**Dampak Kritis:** Risiko tinggi kematian embrionik, infertilitas permanen, dan penurunan vitalitas yang drastis. Populasi dalam bahaya penurunan kualitas genetik permanen."


def contoh_sapi_lengkap() -> pd.DataFrame:
    return pd.DataFrame({
        "Animal_ID": ["PEJANTAN_01", "INDUK_01", "SAPI_C", "SAPI_D", "SAPI_X", "SAPI_B", "SAPI_A", "SAPI_E", "SAPI_F"],
        "Sire_ID": ["-", "-", "PEJANTAN_01", "PEJANTAN_01", "PEJANTAN_01", "SAPI_D", "SAPI_B", "SAPI_B", "SAPI_D"],
        "Dam_ID": ["-", "-", "INDUK_01", "INDUK_01", "INDUK_01", "SAPI_C", "SAPI_C", "-", "-"],
        "Phenotype": [550, 420, 480, 500, 490, 460, 470, 430, 440]
    })


def calculate_breeding_value(res_df: pd.DataFrame, phenotype_col: str, h2: float) -> pd.DataFrame:
    """
    Menghitung Estimated Breeding Value (EBV) sederhana.
    EBV = h^2 * (P - P_avg)
    """
    if phenotype_col not in res_df.columns:
        res_df["EBV"] = 0.0
        return res_df
        
    # Ambil data input saja untuk rata-rata populasi
    input_data = res_df[res_df["Tipe_Data"] == "Data input"]
    if input_data.empty:
        res_df["EBV"] = 0.0
        return res_df
        
    pop_avg = pd.to_numeric(input_data[phenotype_col], errors='coerce').mean()
    
    def calc_ebv(row):
        try:
            p_val = float(row[phenotype_col])
            return h2 * (p_val - pop_avg)
        except:
            return 0.0
            
    res_df["EBV"] = res_df.apply(calc_ebv, axis=1)
    return res_df


def calculate_selection_response(h2: float, sd_p: float, intensity: float) -> float:
    """
    Menghitung Tanggapan Seleksi (R).
    R = i * h^2 * sigma_p
    """
    return intensity * h2 * sd_p


def calculate_inbreeding_depression(f_val: float, depression_rate: float) -> float:
    """
    Menghitung Depresi Inbreeding.
    Penurunan performa = F * Laju Depresi (per 1% F)
    """
    return f_val * depression_rate


def standardize_input(raw_df, id_col, sire_col, dam_col, phenotype_col=None):
    cols = [id_col, sire_col, dam_col]
    if phenotype_col and phenotype_col in raw_df.columns:
        cols.append(phenotype_col)
        df = raw_df[cols].copy()
        df.columns = ["Animal_ID", "Sire_ID", "Dam_ID", "Phenotype"]
    else:
        df = raw_df[cols].copy()
        df.columns = ["Animal_ID", "Sire_ID", "Dam_ID"]

    # Clean IDs but preserve Phenotype if it exists
    for col in df.columns:
        if col != "Phenotype":
            df[col] = df[col].apply(clean_id)
            
    df = df.dropna(subset=["Animal_ID"]).copy()
    if df.empty: raise ValueError("Tidak ada Animal_ID yang valid.")
    return df.reset_index(drop=True)


def klasifikasi_ternak(ebv: float, f_percent: float, dam_id: Optional[str] = None) -> str:
    """Mengklasifikasikan ternak berdasarkan standar pemuliaan."""
    # Deteksi jenis kelamin berdasarkan Dam_ID (jika ada data silsilah lengkap di row)
    # Namun karena ini dipanggil per baris, kita asumsikan klasifikasi utama:
    if ebv > 0.5 and f_percent < 3.125:
        return "Elite Stock"
    elif ebv > 0 and f_percent < 6.25:
        return "Bibit (Breeding Stock)"
    elif ebv > 1.0:
        return "Galur Murni (Line Breeding)"
    elif f_percent > 12.5:
        return "Final Stock (Hanya Potong)"
    else:
        return "Komersial"


def interpretasi_pemuliaan(ebv: float, f_percent: float, depression: float, animal_id: str, results_df: Optional[pd.DataFrame] = None) -> str:
    """Memberikan interpretasi komprehensif untuk nilai pemuliaan dan inbreeding."""
    status_ebv = "Unggul" if ebv > 0 else "Di bawah rata-rata"
    klasifikasi = klasifikasi_ternak(ebv, f_percent)
    
    # Deteksi apakah ini Betina (ada di kolom Dam_ID di populasi)
    is_dam = False
    if results_df is not None:
        is_dam = animal_id in results_df["Dam_ID"].values

    msg = f"**Status Genetik:** {status_ebv} (EBV: {ebv:.2f}). "
    
    if klasi_label := klasifikasi == "Elite Stock":
        if is_dam:
            msg += f"**Klasifikasi:** `Elite Stock (Betina Inti)`. "
            msg += "Rekomendasi: Sangat ideal sebagai indukan inti atau donor embrio (ET). Genetiknya sangat berharga untuk menghasilkan calon pejantan unggul."
        else:
            msg += f"**Klasifikasi:** `Elite Stock (Pejantan Inti)`. "
            msg += "Rekomendasi: Sangat ideal sebagai calon pejantan utama atau sumber semen beku. Pertahankan garis keturunan ini."
    elif klasifikasi == "Bibit (Breeding Stock)":
        msg += f"**Klasifikasi:** `{klasifikasi}`. "
        msg += "Rekomendasi: Cocok sebagai indukan pengganti (replacement) untuk menghasilkan generasi berikutnya."
    elif klasifikasi == "Galur Murni (Line Breeding)":
        msg += f"**Klasifikasi:** `{klasifikasi}`. "
        msg += "Rekomendasi: Potensi genetik luar biasa. Jika inbreeding terkontrol, gunakan untuk fiksasi sifat unggul (Line Breeding)."
    elif klasifikasi == "Final Stock (Hanya Potong)":
        msg += f"**Klasifikasi:** `{klasifikasi}`. "
        msg += "Rekomendasi: Tidak disarankan untuk dikembangbiakkan. Sebaiknya dijadikan ternak potong atau penggemukan karena risiko depresi inbreeding tinggi."
    else:
        msg += f"**Klasifikasi:** `{klasifikasi}`. "
        msg += "Rekomendasi: Layak untuk produksi komersial (susu/daging), namun tidak memiliki nilai genetik istimewa untuk perbaikan populasi."
        
    return msg


def calculate_stats(df: pd.DataFrame):
    """Menghitung korelasi dan regresi antara Inbreeding dan Fenotipe."""
    try:
        # Filter data yang memiliki fenotipe valid
        valid_df = df[df["Phenotype"].apply(lambda x: not is_unknown(x))].copy()
        if len(valid_df) < 3:
            return None
        
        valid_df["Phenotype"] = pd.to_numeric(valid_df["Phenotype"])
        valid_df["F"] = pd.to_numeric(valid_df["Inbreeding_%"])
        
        # Korelasi Pearson
        correlation = valid_df["F"].corr(valid_df["Phenotype"])
        
        # Regresi Linear Sederhana (Y = a + bX)
        x = valid_df["F"]
        y = valid_df["Phenotype"]
        n = len(valid_df)
        
        denominator = (n * (x**2).sum() - (x.sum())**2)
        if denominator == 0: return None

        b = (n * (x * y).sum() - x.sum() * y.sum()) / denominator
        a = (y.sum() - b * x.sum()) / n
        r_squared = correlation**2
        
        return {
            "correlation": correlation,
            "b": b,
            "a": a,
            "r_squared": r_squared
        }
    except:
        return None


def calculate(df_input, h2=0.3, depression_rate=1.0):
    # 1. Identify missing founders
    animal_ids = set(df_input["Animal_ID"])
    parent_ids = set(df_input["Sire_ID"].dropna()).union(set(df_input["Dam_ID"].dropna()))
    missing = sorted(parent_ids - animal_ids)
    
    # 2. Complete the pedigree
    founder_df = pd.DataFrame({"Animal_ID": missing, "Sire_ID": [None]*len(missing), "Dam_ID": [None]*len(missing)})
    if "Phenotype" in df_input.columns:
        founder_df["Phenotype"] = None
        
    df_full = pd.concat([founder_df, df_input], ignore_index=True)
    
    # 3. Build lookup map
    parents_map = {}
    pheno_map = {}
    for row in df_full.itertuples(index=False):
        if row.Animal_ID:
            s_val = None if (row.Sire_ID is None or (isinstance(row.Sire_ID, float) and np.isnan(row.Sire_ID))) else str(row.Sire_ID)
            d_val = None if (row.Dam_ID is None or (isinstance(row.Dam_ID, float) and np.isnan(row.Dam_ID))) else str(row.Dam_ID)
            parents_map[str(row.Animal_ID)] = (s_val, d_val)
            if hasattr(row, 'Phenotype'):
                pheno_map[str(row.Animal_ID)] = row.Phenotype
            
    # 4. Topological Sort
    order = []
    state = {}
    def visit(a):
        if a is None: return
        status = state.get(a, 0)
        if status == 1: raise ValueError(f"Siklus terdeteksi pada {a}")
        if status == 2: return
        state[a] = 1
        s, d = parents_map.get(a, (None, None))
        for p in [s, d]:
            if p is not None: visit(p)
        state[a] = 2
        order.append(a)
    for a in list(parents_map.keys()): visit(a)
    
    # 5. Relationship Matrix (A) Perhitungan - OPTIMIZED FOR NUMPY (Henderson's Method)
    n = len(order)
    idx_map = {a: i for i, a in enumerate(order)}
    A = np.zeros((n, n), dtype=np.float32)
    rows = []
    
    # Pre-calculate population average for EBV
    pop_phenos = [pd.to_numeric(v) for v in pheno_map.values() if v is not None and not is_unknown(v)]
    pop_avg = np.mean(pop_phenos) if pop_phenos else 0
    
    # Map parents to indices once
    parent_indices = []
    for a in order:
        s, d = parents_map.get(a, (None, None))
        si = idx_map.get(s) if s is not None else None
        di = idx_map.get(d) if d is not None else None
        parent_indices.append((si, di))

    # Pre-calculate matrix to avoid slow nested loops for large data
    # Matrix A is symmetric, values only depend on ancestors (already in order)
    for i in range(n):
        si, di = parent_indices[i]
        
        # OFF-DIAGONAL using vector operations for rows 0 to i-1
        if si is not None and di is not None:
            A[i, 0:i] = A[0:i, i] = 0.5 * (A[si, 0:i] + A[di, 0:i])
            F = 0.5 * A[si, di]
        elif si is not None:
            A[i, 0:i] = A[0:i, i] = 0.5 * A[si, 0:i]
            F = 0.0
        elif di is not None:
            A[i, 0:i] = A[0:i, i] = 0.5 * A[di, 0:i]
            F = 0.0
        else:
            F = 0.0
            
        A[i, i] = 1.0 + F
        
        # Data preparation for results
        p_val = pheno_map.get(order[i])
        ebv = 0.0
        depresi = calculate_inbreeding_depression(F * 100, depression_rate)
        
        if p_val is not None and not is_unknown(p_val):
            try:
                ebv = h2 * (float(p_val) - pop_avg)
            except:
                ebv = 0.0

        # HITUNG NILAI HETEROSIS (H)
        # H = P_anak - 0.5 * (P_sire + P_dam)
        heterosis = 0.0
        if si is not None and di is not None and p_val is not None and not is_unknown(p_val):
            p_sire = pheno_map.get(order[si])
            p_dam = pheno_map.get(order[di])
            if p_sire is not None and not is_unknown(p_sire) and p_dam is not None and not is_unknown(p_dam):
                try:
                    p_anak = float(p_val)
                    p_avg_parents = 0.5 * (float(p_sire) + float(p_dam))
                    heterosis = p_anak - p_avg_parents
                except:
                    heterosis = 0.0

        rows.append({
            "Animal_ID": order[i],
            "Sire_ID": parents_map.get(order[i])[0],
            "Dam_ID": parents_map.get(order[i])[1],
            "Phenotype": show_value(p_val),
            "EBV": round(float(ebv), 4),
            "Heterosis": round(float(heterosis), 4),
            "Depresi_Inbreeding": f"-{round(float(depresi), 4)}",
            "Inbreeding_%": round(float(F * 100), 4),
            "Tipe_Data": "Founder tambahan" if order[i] in missing else "Data input"
        })
    
    res_df = pd.DataFrame(rows)

    # Post-process to add classification and interpretation with full population context
    res_df["Klasifikasi"] = res_df.apply(lambda r: klasifikasi_ternak(r["EBV"], r["Inbreeding_%"]), axis=1)
    res_df["Interpretasi_Pemuliaan"] = res_df.apply(
        lambda r: interpretasi_pemuliaan(r["EBV"], r["Inbreeding_%"], float(r["Depresi_Inbreeding"]), r["Animal_ID"], res_df), 
        axis=1
    )
    res_df["Inbreeding_%"] = res_df["Inbreeding_%"].apply(lambda x: round(float(x), 4))
    res_df["Kondisi_Inbreeding"] = res_df["Inbreeding_%"].apply(kondisi_inbreeding)
    res_df["Dampak_Biologis"] = res_df["Inbreeding_%"].apply(dampak_inbreeding)
    res_df["Rekomendasi"] = res_df["Inbreeding_%"].apply(rekomendasi)

    # 6. Deteksi Perkawinan Sedarah (Backcross Bapak-Anak)
    res_df["Peringatan_Reproduksi"] = ""
    for idx, row in res_df.iterrows():
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]
        if sire and dam:
            # Cek apakah bapak adalah bapak dari ibunya (Inbreeding 25% atau lebih)
            grand_sire_of_dam, grand_dam_of_dam = parents_map.get(str(dam), (None, None))
            if str(sire) == str(grand_sire_of_dam):
                res_df.at[idx, "Peringatan_Reproduksi"] = "PERKAWINAN BAPAK-ANAK TERDETEKSI"

    return clean_display(df_input), res_df, pd.DataFrame(A, index=order, columns=order)


def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return pd.read_excel(uploaded_file, dtype=str, keep_default_na=False)


def dot_escape(value): return str(value).replace("\\", "\\\\").replace('"', '\\"')

def make_dot(result_df, max_nodes=50):
    # Batasi visualisasi hanya untuk 50 node pertama agar tidak lag
    df = result_df.head(max_nodes)
    animal_set = set(df["Animal_ID"].astype(str))
    dot = ["digraph Pedigree {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];', "edge [arrowsize=0.6];"]
    
    for _, row in df.iterrows():
        a = dot_escape(row["Animal_ID"])
        f = float(row["Inbreeding_%"])
        klasifikasi = row.get("Klasifikasi", "")
        interpretasi = str(row.get("Interpretasi_Pemuliaan", ""))
        
        # Penentuan Warna & Border
        fill = "#FFFFFF"
        border_color = "black"
        penwidth = "1.0"
        label_suffix = ""
        
        if f >= 25:
            fill = "#FFE4E1" # Merah Muda (High Inbreeding)
        elif f > 0:
            fill = "#FFF4CC" # Kuning (Inbred)
            
        if klasifikasi == "Elite Stock":
            border_color = "#FFD700" # Gold
            penwidth = "3.0"
            if "Betina Inti" in interpretasi:
                label_suffix = "\\nELITE FEMALE"
            else:
                label_suffix = "\\nELITE MALE"
        elif klasifikasi == "Bibit (Breeding Stock)":
            border_color = "#32CD32" # Lime Green
            penwidth = "2.0"
            
        dot.append(f'"{a}" [label="{a}\\nF={f:.2f}%{label_suffix}", fillcolor="{fill}", color="{border_color}", penwidth="{penwidth}"];')
        
    for _, row in df.iterrows():
        a, s, d = dot_escape(row["Animal_ID"]), row["Sire_ID"], row["Dam_ID"]
        if s and str(s) in animal_set: dot.append(f'"{dot_escape(s)}" -> "{a}" [label="sire"];')
        if d and str(d) in animal_set: dot.append(f'"{dot_escape(d)}" -> "{a}" [label="dam"];')
    dot.append("}")
    return "\n".join(dot)


def dots_to_pedigree(result_df):
    """Membuat ringkasan teks cepat (.txt) untuk laporan lapangan."""
    lines = ["LAPORAN ANALISIS PEMULIAAN TERNAK", "="*40]
    lines.append(f"Dicetak pada: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Visualisasi Interaktif: https://inbreeding.streamlit.app/")
    lines.append("-" * 40 + "\n")
    
    for _, row in result_df.iterrows():
        lines.append(f"Individu      : {row['Animal_ID']}")
        lines.append(f"Inbreeding (F): {row['Inbreeding_%']:.2f}%")
        lines.append(f"EBV           : {row['EBV']:.4f}")
        lines.append(f"Klasifikasi   : {row['Klasifikasi']}")
        lines.append(f"Rekomendasi   : {row['Rekomendasi']}")
        lines.append("-" * 30)
    
    lines.append("\n*Catatan: Karena data melebihi 100 ekor, visualisasi pada aplikasi web hanya menampilkan 50 individu pertama untuk menjaga performa.")
    return "\n".join(lines)


def generate_pdf(result_df, settings=None):
    """Membuat laporan PDF profesional dengan detail pemuliaan lengkap."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles['Heading1']
    title_style.alignment = 1 
    
    elements = []
    elements.append(Paragraph("LAPORAN ANALISIS PEMULIAAN TERNAK", title_style))
    elements.append(Spacer(1, 20))
    
    # Summary Statistics PDF
    avg_f = result_df["Inbreeding_%"].mean()
    avg_ebv = result_df["EBV"].mean()
    elements.append(Paragraph(f"<b>Ringkasan Populasi:</b>", styles['Heading2']))
    summary_data = [
        ["Total Populasi", f"{len(result_df)} ekor"],
        ["Rata-rata Inbreeding (F)", f"{avg_f:.2f}%"],
        ["Rata-rata EBV", f"{avg_ebv:.4f}"],
        ["Waktu Analisis", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')],
    ]
    st_table = Table(summary_data, colWidths=[150, 250])
    st_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey)]))
    elements.append(st_table)
    elements.append(Spacer(1, 20))
    
            # Main Data Table
    elements.append(Paragraph("<b>Detail Individu & Klasifikasi:</b>", styles['Heading2']))
    
    # Batasi tabel di PDF hanya 1000 baris pertama untuk performa PDF
    limit_pdf = result_df.head(1000)
    data = [["Animal_ID", "F (%)", "EBV", "Klasifikasi"]]
    
    # Menyiapkan gaya baris (stabilo)
    table_styles = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]

    for i, row in enumerate(limit_pdf.iterrows(), 1):
        _, r = row
        klasifikasi = str(r["Klasifikasi"])
        data.append([
            str(r["Animal_ID"]),
            f"{r['Inbreeding_%']:.2f}",
            f"{r['EBV']:.4f}",
            klasifikasi
        ])
        
        # Tambahkan warna "stabilo" berdasarkan kategori
        if "Elite Stock" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.gold)) # Kuning Emas
        elif "Bibit" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.lightgreen)) # Hijau Muda
        elif "Galur Murni" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.orchid)) # Ungu Muda
        elif "Final Stock" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.lightsalmon)) # Jingga Muda (Coral)
        elif "Komersial" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.lightblue)) # Biru Muda

    if len(result_df) > 1000:
        data.append(["...", "...", "...", "Data lainnya dipangkas"])
    
    t = Table(data, repeatRows=1, colWidths=[100, 80, 80, 150])
    t.setStyle(TableStyle(table_styles))
    elements.append(t)

    # Menambahkan Visualisasi Pedigree ke PDF
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("<b>Bagan Silsilah & Struktur Keturunan:</b>", styles['Heading2']))
    
    # Pendekatan Berbasis Teks (Dot Notation) agar informasi Hubungan tidak hilang 
    # meskipun Graphviz tidak terinstal di server.
    dot_text_style = ParagraphStyle(
        'DotTextStyle',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        leftIndent=20
    )
    
    elements.append(Paragraph("Data hubungan keturunan (30 individu pertama):", styles['Italic']))
    elements.append(Spacer(1, 10))
    
    # Ambil 30 baris silsilah sebagai referensi teks di PDF
    pedigree_lines = []
    for _, row in result_df.head(30).iterrows():
        sire = row['Sire_ID'] if not is_unknown(row['Sire_ID']) else "Unknown"
        dam = row['Dam_ID'] if not is_unknown(row['Dam_ID']) else "Unknown"
        pedigree_lines.append(f"• {row['Animal_ID']} &larr; Sire: {sire}, Dam: {dam}")
    
    pedigree_text = "<br/>".join(pedigree_lines)
    elements.append(Paragraph(pedigree_text, dot_text_style))
    elements.append(Spacer(1, 15))

    # Link URL permanen dengan format yang lebih eksplisit
    link_url = "https://inbreeding.streamlit.app/"
    elements.append(Paragraph(f'<b>Akses Visualisasi Lengkap:</b> <a href="{link_url}" color="blue"><u>{link_url}</u></a>', styles['Normal']))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<i>*Catatan: Karena data melebihi 100 ekor, visualisasi pada aplikasi web hanya menampilkan 50 individu pertama untuk menjaga performa.</i>", styles['Italic']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(
        page_title="Breeding & Inbreeding Analytics",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # CSS for Full Auto Dark Theme (Follows System/Streamlit)
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        
        /* Base Variables (Forces Dark Mode as Default but respects system) */
        :root {
            --bg-main: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f1f5f9;
            --text-sub: #94a3b8;
            --accent-primary: #3b82f6; 
            --accent-secondary: #0ea5e9;
            --border-color: #334155;
            --sidebar-bg: #1e293b;
            --tab-bg: #334155;
            --tab-text: #cbd5e1;
            --header-gradient: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        }

        /* Adaptive Theme based on Streamlit/System */
        @media (prefers-color-scheme: light) {
            :root {
                --bg-main: #f1f5f9;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-sub: #64748b;
                --border-color: #e2e8f0;
                --sidebar-bg: #ffffff;
                --tab-bg: #e2e8f0;
                --tab-text: #475569;
            }
        }

        /* App Container */
        .main { 
            background-color: var(--bg-main) !important;
            font-family: 'Inter', sans-serif;
            color: var(--text-main) !important;
        }

        /* Global Text Color Overrides for Auto Dark */
        .stMarkdown, p, span, label, .stMetric label, [data-testid="stMetricValue"], 
        .stSelectbox label, .stSlider label {
            color: var(--text-main) !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: var(--text-main) !important;
        }

        /* Header Canvas remains dark/gradient for both modes for premium feel */
        .custom-header {
            background: var(--header-gradient);
            padding: 3rem 2rem;
            border-radius: 0 0 2rem 2rem;
            margin: -6rem -4rem 3rem -4rem;
            color: white !important;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
            text-align: center;
        }
        .custom-header * {
            color: white !important;
        }
        
        /* Metrics Styling */
        div[data-testid="stMetric"] {
            background: var(--card-bg) !important;
            padding: 1.5rem;
            border-radius: 1rem;
            border: 1px solid var(--border-color) !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }

        /* Cards & Info Boxes */
        .info-card {
            background: var(--card-bg) !important;
            color: var(--text-main) !important;
            padding: 1.5rem;
            border-radius: 1rem;
            border-left: 5px solid var(--accent-primary);
            border: 1px solid var(--border-color);
            margin: 1rem 0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--border-color) !important;
        }

        /* Tabs Styling */
        .stTabs [data-baseweb="tab"] {
            background-color: var(--tab-bg) !important;
            color: var(--tab-text) !important;
            border-radius: 0.5rem 0.5rem 0 0;
            margin-right: 4px;
        }
        .stTabs [aria-selected="true"] {
            background-color: var(--accent-primary) !important;
            color: white !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
        <div class="custom-header">
            <h1 style="color: white; font-weight: 800; font-size: 3rem; margin-bottom: 0;">GENETIC ANALYTICS</h1>
            <p style="color: #cbd5e1; font-size: 1.2rem; font-weight: 400; letter-spacing: 0.05em;">
                Inbreeding Decision Support System for Advanced Breeding Management
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-header">CONFIGURATION</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Select Data Source",
            ["Contoh sapi lengkap", "Unggah file sendiri"],
            help="Gunakan contoh data untuk mempelajari cara kerja atau unggah file Anda sendiri."
        )
        
        st.info("""
        **Format Data:**
        Pastikan file memiliki kolom:
        - `Animal_ID` (ID Sapi)
        - `Sire_ID` (ID Pejantan)
        - `Dam_ID` (ID Induk)
        
        Gunakan `-` untuk data kosong.
        """)

        if mode == "Contoh sapi lengkap":
            raw_df = contoh_sapi_lengkap()
        else:
            uploaded = st.file_uploader("Unggah CSV atau Excel", type=["csv", "xlsx"])
            if not uploaded:
                st.warning("Silakan unggah file CSV atau Excel untuk memulai.")
                st.stop()
            raw_df = read_file(uploaded)

    # Optimization Warning for Large Data
    if len(raw_df) > 500:
        st.warning(f"Data Besar Terdeteksi ({len(raw_df)} baris): Matriks hubungan aditif sedang dihitung menggunakan akselerasi NumPy. Mohon tunggu sebentar.")

    cols = list(raw_df.columns)
    
    with st.sidebar:
        st.markdown("### Search Pemetaan Kolom")
        id_col = st.selectbox("Kolom Animal_ID", cols, index=0)
        sire_col = st.selectbox("Kolom Sire_ID", cols, index=1 if len(cols)>1 else 0)
        dam_col = st.selectbox("Kolom Dam_ID", cols, index=2 if len(cols)>2 else 0)
        
        phenotype_col = st.selectbox(
            "Kolom Fenotipe (Opsional)", 
            ["-"] + cols, 
            index=cols.index("Phenotype") + 1 if "Phenotype" in cols else 0,
            help="Pilih kolom fenotipe untuk menghitung Nilai Pemuliaan (EBV)."
        )
        pheno_val = None if phenotype_col == "-" else phenotype_col

        st.markdown("### Parameter Genetik")
        h2 = st.slider("Heritabilitas ($h^2$)", 0.0, 1.0, 0.3, 0.05)
        depression_rate = st.slider("Laju Depresi Inbreeding (per 1% F)", 0.0, 5.0, 1.0, 0.1)
        
        st.markdown("### Parameter Seleksi")
        intensity = st.slider("Intensitas Seleksi (i)", 0.0, 3.0, 1.5, 0.1)

    try:
        internal = standardize_input(raw_df, id_col, sire_col, dam_col, pheno_val)
        std_df, res_df, matrix_df = calculate(internal, h2=h2, depression_rate=depression_rate)
        
        res_display_data = res_df[res_df["Tipe_Data"] == "Data input"]
        
        # Define tabs
        tabs = st.tabs(["Hasil & Analisis", "Visualisasi Genetik", "Bagan Pedigree", "Matriks Hubungan (A)", "Heterosis & Crossbreeding"])

        with tabs[0]:
            st.subheader("Ringkasan Data")
            
            # Metrics
            m0, m1, m2, m3, m4 = st.columns(5)
            total_sapi = len(res_display_data)
            inbred_sapi = len(res_display_data[res_display_data["Inbreeding_%"] > 0])
            avg_f = float(res_display_data["Inbreeding_%"].mean())
            max_f = float(res_display_data["Inbreeding_%"].max())
            
            m0.metric("Heritabilitas ($h^2$)", f"{h2:.2f}")
            m1.metric("Total Populasi", f"{total_sapi} ekor")
            m2.metric("Ternak Inbred", f"{inbred_sapi} ekor")
            m3.metric("Rata-rata F", f"{avg_f:.2f}%")
            m4.metric("F Tertinggi", f"{max_f:.2f}%")

            # Klasifikasi Summary
            st.markdown("### Distribusi Klasifikasi Ternak")
            dist = res_display_data["Klasifikasi"].value_counts()
            
            # Cek apakah ada Elite Stock
            has_elite_male = any("Pejantan Inti" in str(x) for x in res_display_data["Interpretasi_Pemuliaan"])
            has_elite_female = any("Betina Inti" in str(x) for x in res_display_data["Interpretasi_Pemuliaan"])
            has_elite = has_elite_male or has_elite_female
            
            c_dist = st.columns(len(dist))
            for i, (label, count) in enumerate(dist.items()):
                # Tampilkan detail gender jika label adalah Elite Stock
                display_label = label
                if label == "Elite Stock":
                    m_count = sum("Pejantan Inti" in str(x) for x in res_display_data["Interpretasi_Pemuliaan"])
                    f_count = sum("Betina Inti" in str(x) for x in res_display_data["Interpretasi_Pemuliaan"])
                    display_label = f"Elite (M:{m_count}, F:{f_count})"
                c_dist[i].metric(display_label, f"{count} ekor")

            # Saran jika tidak ada Elite Stock
            if not has_elite:
                st.warning("Peringatan: Tidak ditemukan Elite Stock dalam populasi ini.")
                with st.container():
                    st.markdown("""
                    <div class="info-card">
                    <b>Saran Perbaikan Genetik:</b><br/>
                    Karena populasi saat ini tidak memiliki individu dengan potensi genetik 'Elite' (EBV tinggi & Inbreeding rendah), berikut langkah yang disarankan:
                    <ol>
                        <li><b>Outcrossing:</b> Datangkan pejantan dari luar (atau semen beku) yang memiliki nilai pemuliaan teruji namun tidak memiliki hubungan kekerabatan dengan indukan saat ini.</li>
                        <li><b>Seleksi Ketat:</b> Gunakan ternak kategori <b>'Bibit (Breeding Stock)'</b> terbaik sebagai indukan pengganti dan hindari penggunaan ternak 'Komersial' sebagai tetua.</li>
                        <li><b>Evaluasi Ulang:</b> Pastikan data fenotipe akurat. Nilai EBV sangat bergantung pada akurasi pencatatan berat/produksi.</li>
                        <li><b>Manajemen Inbreeding:</b> Fokus pada penurunan koefisien inbreeding di bawah 3% untuk membuka peluang munculnya individu Elite di generasi mendatang.</li>
                    </ol>
                    </div>
                    """, unsafe_allow_html=True)

            # Selection Response
            if pheno_val:
                st.markdown("---")
                st.subheader("Estimasi Tanggapan Seleksi")
                r1, r2, r3 = st.columns(3)
                
                # Filter valid phenotypes for SD calculation
                valid_phenos = pd.to_numeric(res_display_data["Phenotype"], errors='coerce').dropna()
                if not valid_phenos.empty:
                    sd_p = valid_phenos.std()
                    avg_p = valid_phenos.mean()
                    response = calculate_selection_response(h2, sd_p, intensity)
                    
                    r1.metric("Rata-rata Fenotipe", f"{avg_p:.2f}")
                    r2.metric("Standar Deviasi Fenotipe", f"{sd_p:.2f}")
                    r1_sub, r2_sub, r3_sub = st.columns(3)
                    r1_sub.metric("Tanggapan Seleksi (R)", f"{response:.4f}")
                    
                    # Heterosis Rata-rata
                    valid_heterosis = res_display_data[res_display_data["Heterosis"] != 0]["Heterosis"]
                    avg_h = valid_heterosis.mean() if not valid_heterosis.empty else 0.0
                    r2_sub.metric("Rata-rata Heterosis", f"{avg_h:.4f}")
                    
                    r3_sub.info(f"Potensi kemajuan genetik per generasi adalah {response:.4f} unit berdasarkan parameter yang dipilih.")
                    
                    # Backcross/Sire-Daughter Alert Section
                    backcross_cases = res_display_data[res_display_data["Peringatan_Reproduksi"] != ""]
                    if not backcross_cases.empty:
                        st.markdown("---")
                        st.error(f"Terdeteksi {len(backcross_cases)} Kasus Perkawinan Bapak-Anak (Backcross)")
                        
                        cols_back = st.columns(2)
                        with cols_back[0]:
                            st.write("**Daftar Individu Terpapar:**")
                            st.dataframe(backcross_cases[["Animal_ID", "Sire_ID", "Dam_ID", "Inbreeding_%"]], hide_index=True)
                        
                        with cols_back[1]:
                            st.markdown("""
                            <div class="info-card" style="border-left: 4px solid #ef4444;">
                            <b>Insight Manajemen Reproduksi & IB:</b><br/>
                            Perkawinan bapak dengan anak kandungnya menghasilkan koefisien inbreeding (F) minimal <b>25%</b>.
                            <ul>
                                <li><b>Risiko Utama:</b> Penurunan vitalitas drastis, risiko cacat bawaan tinggi, dan depresi inbreeding pada performa pertumbuhan/susu.</li>
                                <li><b>Strategi Inseminasi Buatan (IB):</b>
                                    <ul>
                                        <li><b>Database Semen:</b> Inseminator WAJIB mengecek kartu ternak. Jangan gunakan kode semen dari pejantan yang sama dengan Bapak dari betina tersebut.</li>
                                        <li><b>Rotasi Straw:</b> Gunakan straw dari pejantan berbeda <i>lineage</i> (garis keturunan) atau bangsa lain (Crossbreeding) jika diperlukan untuk memutus rantai inbreeding.</li>
                                    </ul>
                                </li>
                                <li><b>Culling:</b> Individu hasil backcross sebaiknya dijadikan <b>Final Stock</b> (ternak potong) dan tidak dipilih sebagai calon indukan (replacement).</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)

                    # Hubungan Inbreeding vs Fenotipe (Korelasi & Regresi)
                    st.markdown("---")
                    st.subheader("Analisis Hubungan Inbreeding vs Fenotipe")
                    stats = calculate_stats(res_display_data)
                    
                    if stats:
                        c_stat1, c_stat2, c_stat3 = st.columns(3)
                        with c_stat1:
                            st.metric("Korelasi (r)", f"{stats['correlation']:.4f}")
                            st.caption("Hubungan antara tingkat inbreeding dan performa.")
                        with c_stat2:
                            st.metric("Regresi (b)", f"{stats['b']:.4f}")
                            st.caption("Penurunan unit fenotipe per 1% kenaikan inbreeding.")
                        with c_stat3:
                            st.metric("Koefisien Determinasi ($R^2$)", f"{stats['r_squared']:.4f}")
                            st.caption("Proporsi variasi fenotipe yang dipengaruhi inbreeding.")
                        
                        st.info(f"**Interpretasi Statis:** Persamaan regresi: $Y = {stats['a']:.2f} + ({stats['b']:.4f}) X$. "
                                f"Artinya, setiap kenaikan 1% inbreeding diprediksi menurunkan fenotipe sebesar {abs(stats['b']):.4f} unit.")
                    else:
                        st.warning("Data tidak cukup untuk menghitung korelasi dan regresi (minimal 3 data dengan fenotipe).")
                else:
                    st.warning("Data fenotipe tidak valid untuk menghitung SD.")

            st.markdown("###  Tabel Hasil Perhitungan")
            # Gunakan st.dataframe dengan height tetap untuk scrolling virtualized
            st.dataframe(clean_display(res_display_data), use_container_width=True, height=500)
            
            # Detailed Interpretation for Selected Animal
            st.markdown("---")
            col_sel1, col_sel2 = st.columns([0.4, 0.6])
            with col_sel1:
                st.subheader(" Interpretasi Individual")
                selected_animal = st.selectbox("Pilih Sapi:", res_display_data["Animal_ID"])
            
            if selected_animal:
                row = res_display_data[res_display_data["Animal_ID"] == selected_animal].iloc[0]
                with col_sel2:
                    st.info(f"**Individu:** {row['Animal_ID']}\n\n{row['Interpretasi_Pemuliaan']}")
                    st.markdown(f"""
                    - **Inbreeding:** {row['Inbreeding_%']}% ({row['Kondisi_Inbreeding']})
                    - **Dampak Performa:** {row['Depresi_Inbreeding']} unit.
                    - **Dampak Biologis:** {row['Dampak_Biologis']}
                    """)
            
            # Additional Information based on Max F
            st.markdown("###  Panduan Interpretasi")
            with st.expander("Klik untuk memahami istilah pemuliaan"):
                st.markdown("""
                - **Koefisien Inbreeding (F):** Persentase kesamaan genetik akibat tetua yang berkerabat.
                - **EBV (Estimated Breeding Value):** Nilai yang menunjukkan potensi genetik yang akan diturunkan ke anak. Semakin tinggi (positif), semakin baik.
                - **Depresi Inbreeding:** Estimasi penurunan performa yang dialami individu karena inbreeding tinggi.
                - **Tanggapan Seleksi (R):** Prediksi kemajuan kualitas populasi pada generasi berikutnya jika seleksi dilakukan.
                """)
            
            st.markdown("###  Analisis Dampak Populasi")
            st.info(dampak_inbreeding(max_f))
            
            # Download options
            st.markdown("###  Unduh Laporan")
            c1, c2, c3 = st.columns(3)
            csv = clean_display(res_df).to_csv(index=False).encode('utf-8')
            c1.download_button(" CSV (Data Lengkap)", csv, "analytics_data.csv", "text/csv", use_container_width=True)
            
            # PDF Report
            pdf_data = generate_pdf(res_display_data)
            c2.download_button(" PDF (Laporan Resmi)", pdf_data, "Breeding_Report.pdf", "application/pdf", use_container_width=True)
            
            # Simple Text Report
            txt_report = dots_to_pedigree(res_display_data)
            c3.download_button(" TXT (Ringkasan Cepat)", txt_report.encode('utf-8'), "summary.txt", "text/plain", use_container_width=True)

            # --- BAGIAN SELEKSI & CULLING ---
            st.markdown("---")
            st.subheader("Rekomendasi Seleksi & Culling")
            
            # Kriteria Seleksi: Top 25% EBV & Inbreeding Rendah
            threshold_ebv = res_display_data["EBV"].quantile(0.75)
            seleksi_df = res_display_data[
                (res_display_data["EBV"] >= threshold_ebv) & 
                (res_display_data["Inbreeding_%"] < 6.25)
            ].sort_values("EBV", ascending=False)
            
            # Kriteria Culling: Inbreeding Sangat Tinggi ATAU Perkawinan Bapak-Anak ATAU EBV sangat rendah (Bottom 10%)
            threshold_low_ebv = res_display_data["EBV"].quantile(0.10)
            culling_df = res_display_data[
                (res_display_data["Inbreeding_%"] >= 25) | 
                (res_display_data["Peringatan_Reproduksi"] != "") |
                (res_display_data["EBV"] <= threshold_low_ebv)
            ].sort_values("Inbreeding_%", ascending=False)
            
            col_sel_recom, col_cul_recom = st.columns(2)
            
            with col_sel_recom:
                st.success(f"**Kandidat Seleksi (Tetua Generasi Berikutnya): {len(seleksi_df)} ekor**")
                st.write("Prioritas berdasarkan EBV tinggi dan risiko inbreeding rendah.")
                st.dataframe(seleksi_df[["Animal_ID", "EBV", "Inbreeding_%", "Klasifikasi"]], hide_index=True)
                
            with col_cul_recom:
                st.error(f"**Kandidat Culling (Ternak Potong/Keluar): {len(culling_df)} ekor**")
                st.write("Berdasarkan inbreeding ekstrem, kasus backcross, atau potensi genetik sangat rendah.")
                st.dataframe(culling_df[["Animal_ID", "EBV", "Inbreeding_%", "Peringatan_Reproduksi"]], hide_index=True)
            # ---------------------------------

            # Extra Insights Section
            st.markdown("---")
            st.markdown("###  Parameter Penting Pemuliaan Ternak")
            
            with st.expander("Lihat Detail Konsep Pemuliaan"):
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.markdown("""
                    **1. Heritabilitas ($h^2$):**
                    Sejauh mana variasi fenotipe disebabkan oleh genetik aditif. 
                    - Rendah (< 0.2): Lingkungan dominan (misal: reproduksi).
                    - Sedang (0.2 - 0.4): Pertumbuhan.
                    - Tinggi (> 0.4): Kualitas karkas/susu.
                    
                    **2. Nilai Pemuliaan (EBV):**
                    Prediksi nilai genetik tetua yang akan diturunkan. EBV adalah dua kali nilai transmisi genetik (*Progeny Difference*).
                    """)
                with col_i2:
                    st.markdown("""
                    **3. Seleksi Tandem:**
                    Metode seleksi satu sifat secara bertahap dalam beberapa generasi sebelum beralih ke sifat lain. Efektif jika fokus pada satu tujuan ekonomi utama.

                    **Strategi Seleksi Tandem & Hubungan Statistik:**
                    Jika Anda menerapkan seleksi tandem, perhatikan langkah strategis berikut:
                    - **Prioritas Sifat:** Mulailah dengan sifat yang memiliki **heritabilitas ($h^2$)** atau nilai ekonomi tertinggi untuk kemajuan awal yang cepat.
                    - **Ambang Batas (Goal):** Tetapkan target minimum sebelum beralih ke sifat berikutnya agar kemajuan genetik pada sifat pertama tidak hilang.
                    - **Analisis Korelasi:** Gunakan metrik **Korelasi ($r$)** di atas untuk memantau apakah perbaikan pada satu sifat tidak merusak sifat lain (korelasi negatif).
                    - **Regresi & Prediksi:** Manfaatkan nilai **Regresi ($b$)** untuk memprediksi sejauh mana performa akan berubah seiring perubahan genetik pada sifat yang sedang diseleksi.

                    **4. Intensitas Seleksi ($i$):**
                    Kekuatan seleksi yang diterapkan. Semakin sedikit ternak yang terpilih sebagai tetua dari total populasi, semakin besar intensitasnya ($i$).
                    """)

            st.markdown("###  Wawasan & Strategi Manajemen")
            
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                st.info("""
                ** Mengapa Inbreeding Berbahaya?**
                Inbreeding meningkatkan peluang munculnya gen resesif yang merugikan. Hal ini menyebabkan:
                - **Penurunan Vitalitas:** Ternak lebih mudah sakit.
                - **Masalah Reproduksi:** Jarak beranak (calving interval) lebih lama.
                - **Penurunan Pertumbuhan:** Berat sapih dan berat dewasa lebih rendah.
                """)
                
            with col_in2:
                st.success("""
                ** Strategi Pencegahan**
                1. **Rotasi Pejantan:** Ganti pejantan setiap 2 tahun atau setelah anak betina pertamanya masuk usia kawin.
                2. **Outcrossing:** Kawinkan ternak dengan individu dari kelompok/populasi lain yang tidak berhubungan.
                3. **Digital Recording:** Gunakan sistem ini secara rutin untuk simulasi sebelum mengawinkan ternak.
                """)

        with tabs[1]:
            st.subheader(" Visualisasi Distribusi Genetik")
            
            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                st.markdown("###  Distribusi Inbreeding (F)")
                # Histogram data F
                f_data = res_display_data["Inbreeding_%"]
                counts, bin_edges = np.histogram(f_data, bins=10)
                hist_df = pd.DataFrame({
                    "Rentang F (%)": [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))],
                    "Jumlah Sapi": counts
                })
                st.bar_chart(hist_df.set_index("Rentang F (%)"))
                st.caption("Grafik ini menunjukkan sebaran tingkat inbreeding dalam populasi.")

            with v_col2:
                st.markdown("###  Potensi Genetik (EBV)")
                # Bar chart for top 10 EBVs
                top_ebv = res_display_data.sort_values("EBV", ascending=False).head(10)
                st.bar_chart(top_ebv.set_index("Animal_ID")["EBV"])
                st.caption("10 ternak dengan Nilai Pemuliaan (EBV) tertinggi.")

            st.markdown("---")
            st.markdown("###  Hubungan Inbreeding vs Fenotipe")
            # Scatter plot using Streamlit's native chart
            # We add a small jitter if needed, but simple scatter 
            scatter_data = res_display_data[res_display_data["Phenotype"].apply(lambda x: not is_unknown(x))].copy()
            if not scatter_data.empty:
                scatter_data["Phenotype_Val"] = pd.to_numeric(scatter_data["Phenotype"])
                st.scatter_chart(
                    scatter_data,
                    x="Inbreeding_%",
                    y="Phenotype_Val",
                    color="Klasifikasi",
                    size="EBV"
                )
                st.caption("Titik-titik menunjukkan hubungan antara Inbreeding (Sumbu X) dan Performa Fenotipe (Sumbu Y). Warna menunjukkan klasifikasi.")
            else:
                st.info("Data fenotipe tidak tersedia untuk visualisasi sebaran.")

        with tabs[2]:
            st.subheader(" Visualisasi Bagan Pedigree")
            if len(res_df) > 100:
                st.info(" **Catatan:** Karena data melebihi 100 ekor, visualisasi hanya menampilkan 50 individu pertama untuk menjaga performa.")
            st.markdown("Bagan ini menunjukkan hubungan keturunan antar sapi. Warna merah menunjukkan inbreeding tinggi (>25%).")
            st.graphviz_chart(make_dot(res_df))

        with tabs[3]:
            st.subheader(" Matriks Hubungan Aditif")
            if len(matrix_df) > 500:
                st.warning(" **Peringatan Performa:** Menampilkan matriks > 500x500 di browser dapat menyebabkan keterlambatan (lag). Disarankan untuk mengunduh CSV jika data sangat besar.")
                if st.button("Tampilkan Matriks Tetap"):
                    st.dataframe(matrix_df, use_container_width=True)
            else:
                st.dataframe(matrix_df, use_container_width=True)

        with tabs[4]:
            st.subheader(" Wawasan Heterosis & Strategi Persilangan")
            
            h_col1, h_col2 = st.columns(2)
            
            with h_col1:
                st.markdown("""
                ###  Apa itu Heterosis?
                **Heterosis** (atau *Hybrid Vigor*) adalah peningkatan performa pada anak hasil persilangan dibandingkan dengan rata-rata kedua tetuanya.
                
                **Rumus Sederhana:**
                $$HF_1 = \\text{Rata-rata Anak} - \\text{Rata-rata Tetua}$$
                $$\\% \\text{Heterosis} = \\frac{HF_1}{\\text{Rata-rata Tetua}} \\times 100\\%$$
                
                **Mengapa Penting?**
                Heterosis sangat efektif untuk sifat-sifat dengan **heritabilitas rendah**, seperti:
                - Kemampuan bertahan hidup (survival).
                - Kesuburan/Reproduksi.
                - Ketahanan terhadap penyakit.
                """)
                
                st.info("""
                ** Insight Penting:**
                Inbreeding adalah kebalikan dari heterosis. Jika inbreeding menurunkan performa (Depresi Inbreeding), maka persilangan (outbreeding) meningkatkannya melalui heterosis.
                """)

            with h_col2:
                st.markdown("""
                ###  Strategi Persilangan (Crossbreeding)
                Untuk mendapatkan heterosis maksimal, Anda dapat menerapkan:
                
                1. **Terminal Cross:** Semua anak hasil persilangan dijual (tidak dijadikan bibit). Memberikan heterosis 100% pada anak.
                2. **Rotational Crossing:** Menggunakan dua atau tiga bangsa secara bergantian. Mempertahankan heterosis sekitar 67% (2 bangsa) atau 86% (3 bangsa) pada generasi berkelanjutan.
                3. **Backcrossing:** Mengawayinkan anak kembali ke salah satu bangsa murni tetuanya. Digunakan untuk memasukkan sifat spesifik dari satu bangsa ke bangsa lain.
                
                **Rekomendasi:**
                - Gunakan pejantan dari bangsa yang unggul pada sifat pertumbuhan.
                - Gunakan induk dari bangsa yang memiliki keunggulan maternal (produksi susu, kesabaran).
                """)

            st.markdown("---")
            st.markdown("""
            ###  Korelasi Genetik vs Heterosis
            Penting untuk diingat bahwa heterosis tidak bersifat permanen (tidak diturunkan secara stabil seperti EBV). 
            - **EBV** digunakan untuk kemajuan genetik jangka panjang (seleksi).
            - **Heterosis** digunakan untuk keuntungan produksi jangka pendek (persilangan).
            
            Sistem kami membantu Anda menjaga agar tingkat inbreeding tetap rendah agar potensi heterosis saat persilangan nanti tetap optimal.
            """)
        
    except Exception as e:
        st.error(f" Terjadi kesalahan dalam pengolahan data: {e}")
        st.exception(e)

    # Footer
    st.markdown("""
        <div class="footer">
            Analytics System v2.5 | Created by Galuh Adi Insani
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
