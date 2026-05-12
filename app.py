import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer


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
        return "🧬 **Dampak:** Variabilitas genetik terjaga maksimal. Tidak ada risiko depresi inbreeding terdeteksi."
    elif percent < 6.25:
        return "🌱 **Dampak:** Pengaruh minimal pada performa. Efek kumulatif mulai muncul jika terjadi terus-menerus dalam beberapa generasi."
    elif percent < 12.5:
        return "⚠️ **Dampak:** Potensi penurunan performa produksi (pertumbuhan, produksi susu) sekitar 2-5%. Sedikit peningkatan risiko penyakit genetik resesif."
    elif percent < 25:
        return "❗ **Dampak:** Depresi inbreeding mulai terlihat nyata. Penurunan fertilitas, daya tahan tubuh melemah, dan potensi munculnya cacat lahir (lethal traits)."
    else:
        return "🚨 **Dampak Kritis:** Risiko tinggi kematian embrionik, infertilitas permanen, dan penurunan vitalitas yang drastis. Populasi dalam bahaya penurunan kualitas genetik permanen."


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
            msg += "👑💎 **Rekomendasi:** Sangat ideal sebagai indukan inti atau donor embrio (ET). Genetiknya sangat berharga untuk menghasilkan calon pejantan unggul."
        else:
            msg += f"**Klasifikasi:** `Elite Stock (Pejantan Inti)`. "
            msg += "👑 **Rekomendasi:** Sangat ideal sebagai calon pejantan utama atau sumber semen beku. Pertahankan garis keturunan ini."
    elif klasifikasi == "Bibit (Breeding Stock)":
        msg += f"**Klasifikasi:** `{klasifikasi}`. "
        msg += "✅ **Rekomendasi:** Cocok sebagai indukan pengganti (replacement) untuk menghasilkan generasi berikutnya."
    elif klasifikasi == "Galur Murni (Line Breeding)":
        msg += f"**Klasifikasi:** `{klasifikasi}`. "
        msg += "🧬 **Rekomendasi:** Potensi genetik luar biasa. Jika inbreeding terkontrol, gunakan untuk fiksasi sifat unggul (Line Breeding)."
    elif klasifikasi == "Final Stock (Hanya Potong)":
        msg += f"**Klasifikasi:** `{klasifikasi}`. "
        msg += "🥩 **Rekomendasi:** Tidak disarankan untuk dikembangbiakkan. Sebaiknya dijadikan ternak potong atau penggemukan karena risiko depresi inbreeding tinggi."
    else:
        msg += f"**Klasifikasi:** `{klasifikasi}`. "
        msg += "🐄 **Rekomendasi:** Layak untuk produksi komersial (susu/daging), namun tidak memiliki nilai genetik istimewa untuk perbaikan populasi."
        
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
    
    # 5. Relationship Matrix (A) Perhitungan
    n = len(order)
    idx_map = {a: i for i, a in enumerate(order)}
    A = np.zeros((n, n))
    rows = []
    
    # Pre-calculate population average for EBV
    pop_phenos = [pd.to_numeric(v) for v in pheno_map.values() if v is not None and not is_unknown(v)]
    pop_avg = np.mean(pop_phenos) if pop_phenos else 0
    
    for i, a in enumerate(order):
        s, d = parents_map.get(a, (None, None))
        si = idx_map.get(s) if s is not None else None
        di = idx_map.get(d) if d is not None else None
        
        # OFF-DIAGONAL: A[i,j] = 0.5(A[si,j] + A[di,j])
        for j in range(i):
            val = 0.0
            if si is not None: val += 0.5 * A[si, j]
            if di is not None: val += 0.5 * A[di, j]
            A[i, j] = A[j, i] = val
            
        # DIAGONAL: A[i,i] = 1 + F_i where F_i = 0.5 * A[si, di]
        F = 0.0
        if si is not None and di is not None:
            F = 0.5 * A[si, di]
            
        A[i, i] = 1.0 + F
        
        # EBV and Depression Calculation
        p_val = pheno_map.get(a)
        ebv = 0.0
        depresi = calculate_inbreeding_depression(F * 100, depression_rate)
        
        if p_val is not None and not is_unknown(p_val):
            try:
                ebv = h2 * (float(p_val) - pop_avg)
            except:
                ebv = 0.0

        rows.append({
            "Animal_ID": a,
            "Sire_ID": s,
            "Dam_ID": d,
            "Phenotype": show_value(p_val),
            "EBV": round(ebv, 4),
            "Depresi_Inbreeding": f"-{round(depresi, 4)}",
            "Inbreeding_%": round(F * 100, 4),
            "Tipe_Data": "Founder tambahan" if a in missing else "Data input"
        })
    
    res_df = pd.DataFrame(rows)

    # Post-process to add classification and interpretation with full population context
    res_df["Klasifikasi"] = res_df.apply(lambda r: klasifikasi_ternak(r["EBV"], r["Inbreeding_%"]), axis=1)
    res_df["Interpretasi_Pemuliaan"] = res_df.apply(
        lambda r: interpretasi_pemuliaan(r["EBV"], r["Inbreeding_%"], float(r["Depresi_Inbreeding"]), r["Animal_ID"], res_df), 
        axis=1
    )
    res_df["Kondisi_Inbreeding"] = res_df["Inbreeding_%"].apply(kondisi_inbreeding)
    res_df["Dampak_Biologis"] = res_df["Inbreeding_%"].apply(dampak_inbreeding)
    res_df["Rekomendasi"] = res_df["Inbreeding_%"].apply(rekomendasi)

    return clean_display(df_input), res_df, pd.DataFrame(A, index=order, columns=order)


def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return pd.read_excel(uploaded_file, dtype=str, keep_default_na=False)


def dot_escape(value): return str(value).replace("\\", "\\\\").replace('"', '\\"')

def make_dot(result_df, max_nodes=150):
    df = result_df.head(max_nodes)
    animal_set = set(df["Animal_ID"].astype(str))
    dot = ["digraph Pedigree {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fontname="Arial"];']
    
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
                label_suffix = "\\n👑💎 ELITE FEMALE"
            else:
                label_suffix = "\\n👑 ELITE MALE"
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
    """Fallback simple text for PDF if no library exists."""
    lines = ["LAPORAN INBREEDING SAPI", "="*30, ""]
    for _, row in result_df.iterrows():
        lines.append(f"Animal: {row['Animal_ID']}")
        lines.append(f"Ref: F={row['Inbreeding_%']}% ({row['Kondisi_Inbreeding']})")
        lines.append(f"Dampak: {row['Dampak_Biologis']}")
        lines.append(f"Rekomendasi: {row['Rekomendasi']}")
        lines.append("-" * 20)
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
    data = [["Animal_ID", "F (%)", "EBV", "Klasifikasi"]]
    for _, row in result_df.iterrows():
        data.append([
            str(row["Animal_ID"]),
            f"{row['Inbreeding_%']:.2f}",
            f"{row['EBV']:.4f}",
            str(row["Klasifikasi"])
        ])
    
    t = Table(data, hAlign='LEFT', colWidths=[120, 80, 80, 150])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]))
    elements.append(t)
    
    doc.build(elements)
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(
        page_title="Breeding & Inbreeding Analytics",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for Professional UI
    st.markdown("""
        <style>
        /* Global Styles */
        .main { background-color: #fcfcfc; }
        h1, h2, h3 { color: #0f172a; font-family: 'Inter', sans-serif; }
        
        /* Metric Styling */
        [data-testid="stMetricValue"] {
            font-size: 2rem !important;
            font-weight: 800 !important;
            color: #2563eb !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #64748b !important;
        }
        
        /* Modern Cards */
        .status-card {
            background: white;
            padding: 24px;
            border-radius: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
            margin-bottom: 1rem;
        }
        .info-card {
            background-color: #f0f9ff;
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #0ea5e9;
            color: #0c4a6e;
            line-height: 1.6;
        }
        
        /* Tabs Customization */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            border-bottom: 2px solid #e2e8f0;
        }
        .stTabs [data-baseweb="tab"] {
            height: 60px;
            font-weight: 600;
            font-size: 1rem;
            color: #64748b;
        }
        .stTabs [aria-selected="true"] {
            color: #2563eb !important;
            border-bottom-color: #2563eb !important;
        }
        
        /* Dark Mode Compatibility */
        @media (prefers-color-scheme: dark) {
            .main { background-color: #0f172a; }
            h1, h2, h3 { color: #f8fafc; }
            .status-card { background: #1e293b; border-color: #334155; }
            .info-card { background-color: #0c4a6e; color: #e0f2fe; border-left-color: #38bdf8; }
            [data-testid="stMetricValue"] { color: #3b82f6 !important; }
        }
        </style>
    """, unsafe_allow_html=True)

    # Header Section
    col_t1, col_t2 = st.columns([0.8, 0.2])
    with col_t1:
        st.title("🧬 Breeding & Inbreeding Analytics")
        st.markdown("""
            **Sistem Pengambilan Keputusan (DSS) Pemuliaan Ternak.**  
            Analisis koefisien inbreeding, nilai pemuliaan (EBV), dan klasifikasi stok bibit berbasis data silsilah dan fenotipe.
        """)
    with col_t2:
        st.image("https://cdn-icons-png.flaticon.com/512/2395/2395796.png", width=100)
    
    st.markdown("---")

    with st.sidebar:
        st.header("⚙️ Konfigurasi")
        mode = st.radio(
            "Pilih Sumber Data",
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

    # Layouting
    tabs = st.tabs(["📊 Hasil & Analisis", "🖇️ Bagan Pedigree", "🔢 Matriks Hubungan (A)"])

    cols = list(raw_df.columns)
    
    with st.sidebar:
        st.markdown("### 🔍 Pemetaan Kolom")
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

        st.markdown("### 🧬 Parameter Genetik")
        h2 = st.slider("Heritabilitas ($h^2$)", 0.0, 1.0, 0.3, 0.05)
        depression_rate = st.slider("Laju Depresi Inbreeding (per 1% F)", 0.0, 5.0, 1.0, 0.1)
        
        st.markdown("### 🎯 Parameter Seleksi")
        intensity = st.slider("Intensitas Seleksi (i)", 0.0, 3.0, 1.5, 0.1)

    try:
        internal = standardize_input(raw_df, id_col, sire_col, dam_col, pheno_val)
        std_df, res_df, matrix_df = calculate(internal, h2=h2, depression_rate=depression_rate)
        
        res_display_data = res_df[res_df["Tipe_Data"] == "Data input"]
        
        with tabs[0]:
            st.subheader("📝 Ringkasan Data")
            
            # Metrics
            m0, m1, m2, m3, m4 = st.columns(5)
            total_sapi = len(res_display_data)
            inbred_sapi = len(res_display_data[res_display_data["Inbreeding_%"] > 0])
            avg_f = float(res_display_data["Inbreeding_%"].mean())
            max_f = float(res_display_data["Inbreeding_%"].max())
            
            m0.metric("Heritabilitas ($h^2$)", f"{h2:.2f}")
            m1.metric("Total Populasi", f"{total_sapi} ekor")
            m2.metric("Sapi Inbred", f"{inbred_sapi} ekor")
            m3.metric("Rata-rata F", f"{avg_f:.2f}%")
            m4.metric("F Tertinggi", f"{max_f:.2f}%")

            # Klasifikasi Summary
            st.markdown("### 🏷️ Distribusi Klasifikasi Ternak")
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
                    display_label = f"Elite (♂:{m_count}, ♀:{f_count})"
                c_dist[i].metric(display_label, f"{count} ekor")

            # Saran jika tidak ada Elite Stock
            if not has_elite:
                st.warning("⚠️ **Peringatan:** Tidak ditemukan **Elite Stock** dalam populasi ini.")
                with st.container():
                    st.markdown("""
                    <div class="info-card">
                    <b>💡 Saran Perbaikan Genetik:</b><br/>
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
                st.subheader("🎯 Estimasi Tanggapan Seleksi")
                r1, r2, r3 = st.columns(3)
                
                # Filter valid phenotypes for SD calculation
                valid_phenos = pd.to_numeric(res_display_data["Phenotype"], errors='coerce').dropna()
                if not valid_phenos.empty:
                    sd_p = valid_phenos.std()
                    response = calculate_selection_response(h2, sd_p, intensity)
                    
                    r1.metric("Standar Deviasi Fenotipe", f"{sd_p:.2f}")
                    r2.metric("Tanggapan Seleksi (R)", f"{response:.4f}")
                    r3.info(f"Potensi kemajuan genetik per generasi adalah {response:.4f} unit berdasarkan parameter yang dipilih.")
                    
                    # Hubungan Inbreeding vs Fenotipe (Korelasi & Regresi)
                    st.markdown("---")
                    st.subheader("📈 Analisis Hubungan Inbreeding vs Fenotipe")
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

            st.markdown("### 📋 Tabel Hasil Perhitungan")
            st.dataframe(clean_display(res_display_data), use_container_width=True, height=400)
            
            # Detailed Interpretation for Selected Animal
            st.markdown("---")
            col_sel1, col_sel2 = st.columns([0.4, 0.6])
            with col_sel1:
                st.subheader("🧐 Interpretasi Individual")
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
            st.markdown("### � Panduan Interpretasi")
            with st.expander("Klik untuk memahami istilah pemuliaan"):
                st.markdown("""
                - **Koefisien Inbreeding (F):** Persentase kesamaan genetik akibat tetua yang berkerabat.
                - **EBV (Estimated Breeding Value):** Nilai yang menunjukkan potensi genetik yang akan diturunkan ke anak. Semakin tinggi (positif), semakin baik.
                - **Depresi Inbreeding:** Estimasi penurunan performa yang dialami individu karena inbreeding tinggi.
                - **Tanggapan Seleksi (R):** Prediksi kemajuan kualitas populasi pada generasi berikutnya jika seleksi dilakukan.
                """)
            
            st.markdown("### �💡 Analisis Dampak Populasi")
            st.info(dampak_inbreeding(max_f))
            
            # Download options
            st.markdown("### 📥 Unduh Laporan")
            c1, c2, c3 = st.columns(3)
            csv = clean_display(res_df).to_csv(index=False).encode('utf-8')
            c1.download_button("📂 CSV (Data Lengkap)", csv, "analytics_data.csv", "text/csv", use_container_width=True)
            
            # PDF Report
            pdf_data = generate_pdf(res_display_data)
            c2.download_button("📄 PDF (Laporan Resmi)", pdf_data, "Breeding_Report.pdf", "application/pdf", use_container_width=True)
            
            # Simple Text Report
            txt_report = dots_to_pedigree(res_display_data)
            c3.download_button("📝 TXT (Ringkasan Cepat)", txt_report.encode('utf-8'), "summary.txt", "text/plain", use_container_width=True)

            # Extra Insights Section
            st.markdown("---")
            st.markdown("### � Parameter Penting Pemuliaan Ternak")
            
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

            st.markdown("### �💡 Wawasan & Strategi Manajemen")
            
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                st.info("""
                **🧬 Mengapa Inbreeding Berbahaya?**
                Inbreeding meningkatkan peluang munculnya gen resesif yang merugikan. Hal ini menyebabkan:
                - **Penurunan Vitalitas:** Ternak lebih mudah sakit.
                - **Masalah Reproduksi:** Jarak beranak (calving interval) lebih lama.
                - **Penurunan Pertumbuhan:** Berat sapih dan berat dewasa lebih rendah.
                """)
                
            with col_in2:
                st.success("""
                **🛡️ Strategi Pencegahan**
                1. **Rotasi Pejantan:** Ganti pejantan setiap 2 tahun atau setelah anak betina pertamanya masuk usia kawin.
                2. **Outcrossing:** Kawinkan ternak dengan individu dari kelompok/populasi lain yang tidak berhubungan.
                3. **Digital Recording:** Gunakan sistem ini secara rutin untuk simulasi sebelum mengawinkan ternak.
                """)

        with tabs[1]:
            st.subheader("🖇️ Visualisasi Bagan Pedigree")
            st.markdown("Bagan ini menunjukkan hubungan keturunan antar sapi. Warna merah menunjukkan inbreeding tinggi (>25%).")
            st.graphviz_chart(make_dot(res_df))

        with tabs[2]:
            st.subheader("🔢 Matriks Hubungan Aditif (Additive Relationship Matrix)")
            st.markdown("Matriks ini menunjukkan nilai $A$ antar individu. Nilai diagonal adalah $1 + F$.")
            st.dataframe(matrix_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"⚠️ Terjadi kesalahan dalam pengolahan data: {e}")
        st.exception(e)

    # Footer
    st.markdown("""
        <div class="footer">
            Developed with ❤️ | Created by Galuh Adi Insani
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
