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
    })


def standardize_input(raw_df, id_col, sire_col, dam_col):
    df = raw_df[[id_col, sire_col, dam_col]].copy()
    df.columns = ["Animal_ID", "Sire_ID", "Dam_ID"]
    for col in ["Animal_ID", "Sire_ID", "Dam_ID"]:
        df[col] = df[col].apply(clean_id)
    df = df.dropna(subset=["Animal_ID"]).copy()
    if df.empty: raise ValueError("Tidak ada Animal_ID yang valid.")
    return df.reset_index(drop=True)


def calculate(df_input):
    # 1. Identify missing founders
    animal_ids = set(df_input["Animal_ID"])
    parent_ids = set(df_input["Sire_ID"].dropna()).union(set(df_input["Dam_ID"].dropna()))
    missing = sorted(parent_ids - animal_ids)
    
    # 2. Complete the pedigree
    founder_df = pd.DataFrame({"Animal_ID": missing, "Sire_ID": [None]*len(missing), "Dam_ID": [None]*len(missing)})
    df_full = pd.concat([founder_df, df_input], ignore_index=True)
    
    # 3. Build lookup map
    # CRITICAL: Convert all IDs to string and handle None/NaN consistently
    parents_map = {}
    for row in df_full.itertuples(index=False):
        if row.Animal_ID:
            s_val = None if (row.Sire_ID is None or (isinstance(row.Sire_ID, float) and np.isnan(row.Sire_ID))) else str(row.Sire_ID)
            d_val = None if (row.Dam_ID is None or (isinstance(row.Dam_ID, float) and np.isnan(row.Dam_ID))) else str(row.Dam_ID)
            parents_map[str(row.Animal_ID)] = (s_val, d_val)
            
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
        
        rows.append({
            "Animal_ID": a,
            "Sire_ID": s,
            "Dam_ID": d,
            "Inbreeding_%": round(F * 100, 4),
            "Kondisi_Inbreeding": kondisi_inbreeding(F * 100),
            "Dampak_Biologis": dampak_inbreeding(F * 100),
            "Rekomendasi": rekomendasi(F * 100),
            "Tipe_Data": "Founder tambahan" if a in missing else "Data input"
        })
    
    res_df = pd.DataFrame(rows)
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
        fill = "#FFE4E1" if f >= 25 else ("#FFF4CC" if f > 0 else "#FFFFFF")
        dot.append(f'"{a}" [label="{a}\\nF={f:.2f}%", fillcolor="{fill}"];')
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


def generate_pdf(result_df):
    """Membuat laporan PDF profesional menggunakan ReportLab."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles['Heading1']
    title_style.alignment = 1  # Center
    
    elements = []
    
    # Judul
    elements.append(Paragraph("LAPORAN ANALISIS INBREEDING TERNAK", title_style))
    elements.append(Spacer(1, 12))
    
    # Ringkasan Ringkas
    avg_f = result_df["Inbreeding_%"].mean()
    max_f = result_df["Inbreeding_%"].max()
    summary_text = f"Total Ternak: {len(result_df)} | Rata-rata Inbreeding: {avg_f:.2f}% | Inbreeding Maksimum: {max_f:.2f}%"
    elements.append(Paragraph(summary_text, styles['Normal']))
    elements.append(Spacer(1, 20))
    
    # Tabel Data
    data = [["Animal_ID", "Sire", "Dam", "F (%)", "Indikasi"]]
    for _, row in result_df.iterrows():
        data.append([
            str(row["Animal_ID"]),
            str(row["Sire_ID"]),
            str(row["Dam_ID"]),
            f"{row['Inbreeding_%']:.2f}",
            str(row["Kondisi_Inbreeding"])
        ])
    
    t = Table(data, hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.dodgerblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    elements.append(t)
    elements.append(Spacer(1, 25))
    
    # Bagian Pencegahan & Insight
    elements.append(Paragraph("STRATEGI PENCEGAHAN & MANAJEMEN", styles['Heading2']))
    elements.append(Spacer(1, 10))
    
    insight_text = """
    <b>1. Pertukaran Pejantan (Sire Rotation):</b> Hindari penggunaan pejantan yang sama selama lebih dari dua generasi pada kelompok induk yang sama. Tukar pejantan dengan peternak lain atau gunakan semen beku dari garis keturunan berbeda.<br/><br/>
    <b>2. Pencatatan Pedigree yang Ketat:</b> Inbreeding hanya bisa dicegah jika silsilah ternak tercatat dengan baik. Pastikan setiap kelahiran dicatat induk dan bapaknya.<br/><br/>
    <b>3. Batas Toleransi:</b> Usahakan koefisien inbreeding tetap di bawah 6.25%. Jika sudah mencapai 12.5%, segera lakukan 'outcrossing' (perkawinan dengan individu yang tidak berkerabat sama sekali).<br/><br/>
    <b>4. Seleksi Berbasis Nilai Pemuliaan:</b> Jangan hanya memilih ternak berdasarkan fisik, tapi juga perhatikan hubungan kekerabatannya untuk mencegah depresi inbreeding.
    """
    elements.append(Paragraph(insight_text, styles['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer


def main():
    st.set_page_config(
        page_title="Kalkulator Inbreeding Ternak",
        page_icon="🐄",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 1.8rem !important;
            font-weight: 700 !important;
            color: #1e3a8a !important; /* Biru gelap untuk kontras tinggi */
        }
        [data-testid="stMetricLabel"] {
            font-size: 1rem !important;
            font-weight: 600 !important;
            color: #374151 !important; /* Abu-abu gelap */
        }
        /* Penyesuaian khusus untuk Mode Gelap agar tetap terbaca */
        @media (prefers-color-scheme: dark) {
            [data-testid="stMetricValue"] {
                color: #60a5fa !important; /* Biru terang untuk mode gelap */
            }
            [data-testid="stMetricLabel"] {
                color: #e5e7eb !important; /* Abu-abu sangat terang */
            }
        }
        .stMetric {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 10px;
            border: 1px solid rgba(128, 128, 128, 0.2);
        }
        h1, h2, h3 {
            color: #1e3a8a;
        }
        @media (prefers-color-scheme: dark) {
            h1, h2, h3 {
                color: #60a5fa;
            }
        }
        /* Footer style */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: transparent;
            color: #6b7280;
            text-align: center;
            padding: 10px;
            font-size: 0.8rem;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🐄 Kalkulator Inbreeding Ternak")
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

    try:
        internal = standardize_input(raw_df, id_col, sire_col, dam_col)
        std_df, res_df, matrix_df = calculate(internal)
        
        res_display_data = res_df[res_df["Tipe_Data"] == "Data input"]
        
        with tabs[0]:
            st.subheader("📝 Ringkasan Data")
            
            # Metrics
            m1, m2, m3, m4 = st.columns(4)
            total_sapi = len(res_display_data)
            inbred_sapi = len(res_display_data[res_display_data["Inbreeding_%"] > 0])
            avg_f = float(res_display_data["Inbreeding_%"].mean())
            max_f = float(res_display_data["Inbreeding_%"].max())
            
            m1.metric("Total Populasi", f"{total_sapi} ekor")
            m2.metric("Sapi Inbred", f"{inbred_sapi} ekor")
            m3.metric("Rata-rata F", f"{avg_f:.2f}%")
            m4.metric("F Tertinggi", f"{max_f:.2f}%")

            st.markdown("### 📋 Tabel Hasil Perhitungan")
            st.dataframe(clean_display(res_display_data), use_container_width=True, height=400)
            
            # Additional Information based on Max F
            st.markdown("### 💡 Analisis Dampak Populasi")
            st.info(dampak_inbreeding(max_f))
            
            # Download options
            st.markdown("### 📥 Unduh Hasil")
            c1, c2, c3 = st.columns(3)
            csv = clean_display(res_df).to_csv(index=False).encode('utf-8')
            c1.download_button("📂 CSV", csv, "hasil_inbreeding.csv", "text/csv", use_container_width=True)
            
            # PDF Report
            pdf_data = generate_pdf(res_display_data)
            c2.download_button("📄 Laporan PDF", pdf_data, "Laporan_Inbreeding.pdf", "application/pdf", use_container_width=True)
            
            # Simple Text Report
            txt_report = dots_to_pedigree(res_display_data)
            c3.download_button("📝 Ringkasan TXT", txt_report.encode('utf-8'), "laporan_inbreeding.txt", "text/plain", use_container_width=True)

            # Extra Insights Section
            st.markdown("---")
            st.markdown("### 💡 Wawasan & Strategi Manajemen")
            
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
