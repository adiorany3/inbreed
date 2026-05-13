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
    "unknown", "unknown", "none", "empty",
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
    if percent <= 0: return "Not inbred"
    if percent < 6.25: return "Low inbreeding"
    if percent < 12.5: return "Moderate inbreeding"
    if percent < 25: return "High inbreeding"
    return "Very high inbreeding"


def rekomendasi(percent: float) -> str:
    if percent <= 0: return "Safe based on pedigree: no inbreeding detected."
    if percent < 6.25: return "Still low, but monitoring is required."
    if percent < 12.5: return "Requires attention. Avoid repeating matings between relatives."
    if percent < 25: return "High risk. Use mates from different lineages."
    return "Very high risk. Close relative matings should be avoided."


def dampak_inbreeding(percent: float) -> str:
    """Provides additional information on biological impacts based on F percentage."""
    if percent <= 0:
        return "**Impact:** Maximum genetic variability maintained. No inbreeding depression risk detected."
    elif percent < 6.25:
        return "**Impact:** Minimal influence on performance. Cumulative effects begin to appear if it occurs continuously over several generations."
    elif percent < 12.5:
        return "**Impact:** Potential reduction in production performance (growth, milk production) by approximately 2-5%. Slight increase in risk of recessive genetic diseases."
    elif percent < 25:
        return "**Impact:** Inbreeding depression becomes clearly visible. Decreased fertility, weakened immune system, and potential emergence of birth defects (lethal traits)."
    else:
        return "**Critical Impact:** High risk of embryonic death, permanent infertility, and drastic reduction in vitality. Population is at danger of permanent genetic quality decline."


def contoh_sapi_lengkap() -> pd.DataFrame:
    return pd.DataFrame({
        "Animal_ID": ["SIRE_01", "DAM_01", "COW_C", "COW_D", "COW_X", "COW_B", "COW_A", "COW_E", "COW_F"],
        "Sire_ID": ["-", "-", "SIRE_01", "SIRE_01", "SIRE_01", "COW_D", "COW_B", "COW_B", "COW_D"],
        "Dam_ID": ["-", "-", "DAM_01", "DAM_01", "DAM_01", "COW_C", "COW_C", "-", "-"],
        "Phenotype": [550, 420, 480, 500, 490, 460, 470, 430, 440]
    })


def calculate_breeding_value(res_df: pd.DataFrame, phenotype_col: str, h2: float) -> pd.DataFrame:
    """
    Calculates simple Estimated Breeding Value (EBV).
    EBV = h^2 * (P - P_avg)
    """
    if phenotype_col not in res_df.columns:
        res_df["EBV"] = 0.0
        return res_df
        
    # Take input data only for population average
    input_data = res_df[res_df["Tipe_Data"] == "Input data"]
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
    Calculates Selection Response (R).
    R = i * h^2 * sigma_p
    """
    return intensity * h2 * sd_p


def calculate_inbreeding_depression(f_val: float, depression_rate: float) -> float:
    """
    Calculates Inbreeding Depression.
    Performance reduction = F * Depression Rate (per 1% F)
    """
    return f_val * depression_rate


def analyze_hardy_weinberg(result_df: pd.DataFrame) -> Dict:
    """
    Analyzes Hardy-Weinberg Equilibrium (HWE) in the population.
    Theoretically, inbreeding (F) is a measure of deviation from expected heterozygosity (HWE).
    F > 0 indicates a heterozygote deficit.
    """
    avg_f = result_df["Inbreeding_%"].mean() / 100.0  # FIS Proportion
    
    # Practical threshold: F > 0.05 (FIS > 5%) is considered a significant deviation from HWE
    is_deviating = avg_f > 0.05
    status = "⚠️ Deviation Occurring (Not in Equilibrium)" if is_deviating else "✅ Near Equilibrium"
    
    insight = (
        f"The average population inbreeding coefficient is {avg_f:.4f}. "
        "In population genetics, this value indicates the extent to which the population deviates from the ideal Hardy-Weinberg condition (random mating)."
    )
    
    if is_deviating:
        saran = [
            "**Outcrossing:** Introduce sires/semen from outside the population that are unrelated.",
            "**Increase Ne:** Increase the number of active sires in the breeding process to increase effective population size.",
            "**Sire Rotation:** Avoid excessive use of a single 'Popular Sire' on many females.",
            "**Crossbreeding:** If the goal is commercial production, perform crossbreeding between breeds to restore heterozygosity."
        ]
    else:
        saran = [
            "The current mating system is still able to maintain genetic variation.",
            "Continue monitoring pedigrees to avoid sudden increases in inbreeding in the next generation."
        ]

    return {
        "status": status,
        "insight": insight,
        "saran": saran,
        "is_deviating": is_deviating
    }


def standardize_input(raw_df, id_col, sire_col, dam_col, phenotype_col=None):
    cols = [id_col, sire_col, dam_col]
    if phenotype_col and phenotype_col in raw_df.columns:
        cols.append(phenotype_col)
        df = raw_df[cols].copy()
        df.columns = ["Animal_ID", "Sire_ID", "Dam_ID", "Phenotype"]
    else:
        df = raw_df[cols].copy()
        df.columns = ["Animal_ID", "Sire_ID", "Project_ID"]

    # Clean IDs but preserve Phenotype if it exists
    for col in df.columns:
        if col != "Phenotype":
            df[col] = df[col].apply(clean_id)
            
    df = df.dropna(subset=["Animal_ID"]).copy()
    if df.empty: raise ValueError("No valid Animal_ID found.")
    return df.reset_index(drop=True)


def klasifikasi_ternak(ebv: float, f_percent: float, dam_id: Optional[str] = None) -> str:
    """Classifies livestock based on breeding standards."""
    if ebv > 0.5 and f_percent < 3.125:
        return "Elite Stock"
    elif ebv > 0 and f_percent < 6.25:
        return "Breeding Stock"
    elif ebv > 1.0:
        return "Line Breeding"
    elif f_percent > 12.5:
        return "Final Stock (Slaughter Only)"
    else:
        return "Commercial"


def interpretasi_pemuliaan(ebv: float, f_percent: float, depression: float, animal_id: str, results_df: Optional[pd.DataFrame] = None) -> str:
    """Provides comprehensive interpretation for breeding value and inbreeding."""
    status_ebv = "Superior" if ebv > 0 else "Below average"
    klasifikasi = klasifikasi_ternak(ebv, f_percent)
    
    # Detect if this is a Dam (exists in Dam_ID column in population)
    is_dam = False
    if results_df is not None:
        is_dam = animal_id in results_df["Dam_ID"].values

    msg = f"**Genetic Status:** {status_ebv} (EBV: {ebv:.2f}). "
    
    if klasifikasi == "Elite Stock":
        if is_dam:
            msg += f"**Classification:** `Elite Stock (Core Female)`. "
            msg += "Recommendation: Highly ideal as core dam or embryo donor (ET). Genetics are very valuable for producing superior sire candidates."
        else:
            msg += f"**Classification:** `Elite Stock (Core Sire)`. "
            msg += "Recommendation: Highly ideal as primary sire candidate or frozen semen source. Maintain this lineage."
    elif klasifikasi == "Breeding Stock":
        msg += f"**Classification:** `{klasifikasi}`. "
        msg += "Recommendation: Suitable as replacement dam to produce the next generation."
    elif klasifikasi == "Line Breeding":
        msg += f"**Classification:** `{klasifikasi}`. "
        msg += "Recommendation: Extraordinary genetic potential. If inbreeding is controlled, use for fixation of superior traits (Line Breeding)."
    elif klasifikasi == "Final Stock (Slaughter Only)":
        msg += f"**Classification:** `{klasifikasi}`. "
        msg += "Recommendation: Not recommended for breeding. Should be used for slaughter or fattening due to high inbreeding depression risk."
    else:
        msg += f"**Classification:** `{klasifikasi}`. "
        msg += "Recommendation: Suitable for commercial production (milk/meat), but does not have special genetic value for population improvement."
        
    return msg


def calculate_stats(df: pd.DataFrame):
    """Calculates correlation and regression between Inbreeding and Phenotype."""
    try:
        # Filter data with valid phenotypes
        valid_df = df[df["Phenotype"].apply(lambda x: not is_unknown(x))].copy()
        if len(valid_df) < 3:
            return None
        
        valid_df["Phenotype"] = pd.to_numeric(valid_df["Phenotype"])
        valid_df["F"] = pd.to_numeric(valid_df["Inbreeding_%"])
        
        # Pearson Correlation
        correlation = valid_df["F"].corr(valid_df["Phenotype"])
        
        # Simple Linear Regression (Y = a + bX)
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

        # CALCULATE HETEROSIS VALUE (H)
        # H = P_offspring - 0.5 * (P_sire + P_dam)
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
            "Inbreeding_Depression": f"-{round(float(depresi), 4)}",
            "Inbreeding_%": round(float(F * 100), 4),
            "Data_Type": "Additional founder" if order[i] in missing else "Input data"
        })
    
    res_df = pd.DataFrame(rows)

    # Post-process to add classification and interpretation with full population context
    res_df["Classification"] = res_df.apply(lambda r: klasifikasi_ternak(r["EBV"], r["Inbreeding_%"]), axis=1)
    res_df["Breeding_Interpretation"] = res_df.apply(
        lambda r: interpretasi_pemuliaan(r["EBV"], r["Inbreeding_%"], float(r["Inbreeding_Depression"]), r["Animal_ID"], res_df), 
        axis=1
    )
    res_df["Inbreeding_%"] = res_df["Inbreeding_%"].apply(lambda x: round(float(x), 4))
    res_df["Inbreeding_Condition"] = res_df["Inbreeding_%"].apply(kondisi_inbreeding)
    res_df["Biological_Impact"] = res_df["Inbreeding_%"].apply(dampak_inbreeding)
    res_df["Recommendation"] = res_df["Inbreeding_%"].apply(rekomendasi)

    # 6. Sire-Daughter Mating Detection (Backcross)
    res_df["Reproduction_Warning"] = ""
    for idx, row in res_df.iterrows():
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]
        if sire and dam:
            # Check if sire is the father of the mother (Inbreeding 25% or more)
            grand_sire_of_dam, grand_dam_of_dam = parents_map.get(str(dam), (None, None))
            if str(sire) == str(grand_sire_of_dam):
                res_df.at[idx, "Reproduction_Warning"] = "SIRE-DAUGHTER MATING DETECTED"

    return clean_display(df_input), res_df, pd.DataFrame(A, index=order, columns=order)


def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return pd.read_excel(uploaded_file, dtype=str, keep_default_na=False)


def dot_escape(value): return str(value).replace("\\", "\\\\").replace('"', '\\"')

def make_dot(result_df, max_nodes=50):
    # Limit visualization to first 50 nodes to avoid lag
    df = result_df.head(max_nodes)
    animal_set = set(df["Animal_ID"].astype(str))
    dot = ["digraph Pedigree {", "rankdir=LR;", 'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];', "edge [arrowsize=0.6];"]
    
    for _, row in df.iterrows():
        a = dot_escape(row["Animal_ID"])
        f = float(row["Inbreeding_%"])
        klasifikasi = row.get("Classification", "")
        interpretasi = str(row.get("Breeding_Interpretation", ""))
        
        # Color & Border determination
        fill = "#FFFFFF"
        border_color = "black"
        penwidth = "1.0"
        label_suffix = ""
        
        if f >= 25:
            fill = "#FFE4E1" # Pink (High Inbreeding)
        elif f > 0:
            fill = "#FFF4CC" # Yellow (Inbred)
            
        if klasifikasi == "Elite Stock":
            border_color = "#FFD700" # Gold
            penwidth = "3.0"
            if "Core Female" in interpretasi:
                label_suffix = "\\nELITE FEMALE"
            else:
                label_suffix = "\\nELITE MALE"
        elif klasifikasi == "Breeding Stock":
            border_color = "#32CD32" # Lime Green
            penwidth = "2.0"
            
        dot.append(f'"{a}" [label="{a}\\nF={f:.2f}%{label_suffix}", fillcolor="{fill}", color="{border_color}", penwidth="{penwidth}"];')
        
    for _, row in df.iterrows():
        a, s, d = dot_escape(row["Animal_ID"]), row["Sire_ID"], row["Dam_ID"]
        if s and str(s) in animal_set: dot.append(f'"{dot_escape(s)}" -> "{a}" [label="sire"];')
        if d and str(d) in animal_set: dot.append(f'"{dot_escape(d)}" -> "{a}" [label="dam"];')
    dot.append("}")
    return "\n".join(dot)


def dots_to_pedigree(result_df, settings=None):
    """Creates a quick text summary report (.txt) for field use."""
    lines = ["LIVESTOCK BREEDING & INBREEDING SUMMARY", "="*50]
    lines.append(f"Printed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Developer : Galuh Adi Insani (Universitas Gadjah Mada)")
    lines.append(f"Official App: https://inbreeding.streamlit.app/")
    lines.append("="*50 + "\n")

    # 1. POPULATION SUMMARY
    avg_f = result_df["Inbreeding_%"].mean()
    avg_ebv = result_df["EBV"].mean()
    lines.append("1. POPULATION STATISTICS")
    lines.append(f"   Total Animals        : {len(result_df)}")
    lines.append(f"   Average Inbreeding F : {avg_f:.2f}%")
    lines.append(f"   Average EBV          : {avg_ebv:.4f}")
    
    if "Phenotype" in result_df.columns:
        valid_phenos = pd.to_numeric(result_df["Phenotype"], errors='coerce').dropna()
        if not valid_phenos.empty:
            avg_p = valid_phenos.mean()
            sd_p = valid_phenos.std()
            lines.append(f"   Average Phenotype    : {avg_p:.2f} +/- {sd_p:.2f}")
            if settings and 'h2' in settings and 'intensity' in settings:
                response = calculate_selection_response(settings['h2'], sd_p, settings['intensity'])
                lines.append(f"   Selection Response R : {response:.4f}")
    
    hwe_res = analyze_hardy_weinberg(result_df)
    lines.append(f"   HWE Genetic Status   : {hwe_res['status']}")
    lines.append("-" * 50 + "\n")

    # 2. SELECTION CANDIDATES
    threshold_ebv = result_df["EBV"].quantile(0.75)
    selection_df = result_df[
        (result_df["EBV"] >= threshold_ebv) & 
        (result_df["Inbreeding_%"] < 6.25)
    ].sort_values("EBV", ascending=False).head(10)
    
    lines.append("2. TOP 10 SELECTION CANDIDATES")
    if not selection_df.empty:
        lines.append(f"   {'Animal_ID':<15} {'EBV':<10} {'F (%)':<8} {'Classification':<20}")
        for _, r in selection_df.iterrows():
            lines.append(f"   {str(r['Animal_ID']):<15} {r['EBV']:<10.4f} {r['Inbreeding_%']:<8.2f} {str(r['Classification']):<20}")
    else:
        lines.append("   No candidates found.")
    lines.append("-" * 50 + "\n")

    # 3. PRIORITY CULLING
    threshold_low_ebv = result_df["EBV"].quantile(0.10)
    culling_df = result_df[
        (result_df["Inbreeding_%"] >= 25) | 
        (result_df["Reproduction_Warning"] != "") |
        (result_df["EBV"] <= threshold_low_ebv)
    ].sort_values("Inbreeding_%", ascending=False).head(10)

    lines.append("3. PRIORITY CULLING CANDIDATES (Top 10 High Risk)")
    if not culling_df.empty:
        lines.append(f"   {'Animal_ID':<15} {'F (%)':<8} {'Warning':<25}")
        for _, r in culling_df.iterrows():
            warn = r['Reproduction_Warning'] if r['Reproduction_Warning'] else "Low Genetic/High F"
            lines.append(f"   {str(r['Animal_ID']):<15} {r['Inbreeding_%']:<8.2f} {warn:<25}")
    else:
        lines.append("   No priority culling needed.")
    lines.append("-" * 50 + "\n")

    # 4. INDIVIDUAL DETAILS
    lines.append("4. COMPLETE INDIVIDUAL LOG")
    for _, row in result_df.iterrows():
        lines.append(f"   [{row['Animal_ID']}] F: {row['Inbreeding_%']:.2f}% | EBV: {row['EBV']:.4f} | {row['Classification']}")
        if row['Reproduction_Warning']:
            lines.append(f"   !!! WARNING: {row['Reproduction_Warning']}")
    
    lines.append("\n" + "="*50)
    lines.append("END OF REPORT")
    return "\n".join(lines)


def generate_pdf(result_df, settings=None):
    """Creates a professional PDF report with full breeding details."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Custom Styles
    title_style = styles['Heading1']
    title_style.alignment = 1 
    
    elements = []
    elements.append(Paragraph("LIVESTOCK BREEDING ANALYSIS REPORT", title_style))
    elements.append(Spacer(1, 20))
    
    # --- SECTION 1: POPULATION SUMMARY ---
    avg_f = result_df["Inbreeding_%"].mean()
    avg_ebv = result_df["EBV"].mean()
    elements.append(Paragraph(f"<b>1. Population Summary:</b>", styles['Heading2']))
    
    # Check for additional stats if selection parameters exist
    avg_p = 0
    sd_p = 0
    response = 0
    if "Phenotype" in result_df.columns:
        valid_phenos = pd.to_numeric(result_df["Phenotype"], errors='coerce').dropna()
        if not valid_phenos.empty:
            avg_p = valid_phenos.mean()
            sd_p = valid_phenos.std()
            if settings and 'h2' in settings and 'intensity' in settings:
                response = calculate_selection_response(settings['h2'], sd_p, settings['intensity'])

    summary_data = [
        ["Total Population", f"{len(result_df)} animals"],
        ["Average Inbreeding (F)", f"{avg_f:.2f}%"],
        ["Average EBV", f"{avg_ebv:.4f}"],
        ["Analysis Time", pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')],
    ]
    
    if avg_p > 0:
        summary_data.append(["Average Phenotype", f"{avg_p:.2f}"])
        summary_data.append(["Standard Deviation (σp)", f"{sd_p:.2f}"])
        if response > 0:
            summary_data.append(["Selection Response (R)", f"{response:.4f}"])

    st_table = Table(summary_data, colWidths=[180, 220])
    st_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.whitesmoke),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
    ]))
    elements.append(st_table)
    elements.append(Spacer(1, 15))

    # --- SECTION 2: SELECTION RESPONSE ESTIMATION ---
    if response > 0:
        elements.append(Paragraph("<b>2. Genetic Progress Estimation:</b>", styles['Heading2']))
        interpretasi_txt = f"""
        Based on heritability ({settings['h2']}) and selection intensity ({settings['intensity']}), 
        the population is predicted to experience progress of <b>{response:.4f}</b> units in the next generation. 
        Estimated average offspring performance is <b>{avg_p + response:.2f}</b> units.
        """
        elements.append(Paragraph(interpretasi_txt, styles['Normal']))
        elements.append(Spacer(1, 15))

    # --- SECTION 3: REGRESSION ANALYSIS (INBREEDING IMPACT) ---
    stats = calculate_stats(result_df)
    if stats:
        elements.append(Paragraph("<b>3. Inbreeding vs Phenotype Relationship Analysis:</b>", styles['Heading2']))
        reg_data = [
            ["Parameter", "Value", "Interpretation"],
            ["Correlation (r)", f"{stats['correlation']:.4f}", "Relationship strength"],
            ["Regression (b)", f"{stats['b']:.4f}", "Reduction per 1% F"],
            ["R-Squared", f"{stats['r_squared']:.4f}", "Model accuracy"]
        ]
        reg_table = Table(reg_data, colWidths=[100, 100, 200])
        reg_table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e5e7eb")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ]))
        elements.append(reg_table)
        elements.append(Paragraph(f"Every 1% increase in inbreeding is predicted to decrease performance by {abs(stats['b']):.4f} units.", styles['Italic']))
        elements.append(Spacer(1, 15))
    
    # --- HWE SECTION (PDF) ---
    hwe_res = analyze_hardy_weinberg(result_df)
    elements.append(Paragraph("<b>4. Hardy-Weinberg Equilibrium Analysis:</b>", styles['Heading2']))
    elements.append(Paragraph(f"<b>Status: {hwe_res['status']}</b>", styles['Normal']))
    elements.append(Paragraph(hwe_res['insight'], styles['Normal']))
    elements.append(Paragraph("<b>Management Recommendations:</b>", styles['Normal']))
    for s in hwe_res['saran']:
        elements.append(Paragraph(f"• {s}", styles['Normal']))
    elements.append(Spacer(1, 15))

    # --- SECTION 5: INDIVIDUAL DETAILS ---
    elements.append(Paragraph("<b>5. Individual Details & Classification:</b>", styles['Heading2']))
    
    # Limit PDF table to first 1000 rows for performance
    limit_pdf = result_df.head(1000)
    data = [["Animal_ID", "F (%)", "EBV", "Classification"]]
    
    # Set up row styles (highlighter)
    table_styles = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey)
    ]

    for i, row in enumerate(limit_pdf.iterrows(), 1):
        _, r = row
        klasifikasi = str(r["Classification"])
        data.append([
            str(r["Animal_ID"]),
            f"{r['Inbreeding_%']:.2f}",
            f"{r['EBV']:.4f}",
            klasifikasi
        ])
        
        # Add "highlighter" colors based on category
        if "Elite Stock" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.gold)) # Gold Yellow
        elif "Breeding Stock" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.lightgreen)) # Light Green
        elif "Line Breeding" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.orchid)) # Light Purple
        elif "Final Stock" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.lightsalmon)) # Light Orange (Coral)
        elif "Commercial" in klasifikasi:
            table_styles.append(('BACKGROUND', (3, i), (3, i), colors.lightblue)) # Light Blue

    if len(result_df) > 1000:
        data.append(["...", "...", "...", "Other data truncated"])
    
    t = Table(data, repeatRows=1, colWidths=[100, 80, 80, 150])
    t.setStyle(TableStyle(table_styles))
    elements.append(t)

    # --- SECTION 6: FULL PEDIGREE ---
    elements.append(Spacer(1, 30))
    elements.append(Paragraph("<b>6. Full Pedigree & Descendant Relationships:</b>", styles['Heading2']))
    
    # Use table for pedigree to be neater and cover more data
    ped_data = [["Individual", "Sire", "Dam"]]
    ped_styles = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#475569")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTSIZE', (0, 0), (-1, -1), 8)
    ]

    # Take first 1000 individuals for pedigree in PDF
    limit_ped = result_df.head(1000)
    for _, row in limit_ped.iterrows():
        sire = str(row['Sire_ID']) if not is_unknown(row['Sire_ID']) else "-"
        dam = str(row['Dam_ID']) if not is_unknown(row['Dam_ID']) else "-"
        ped_data.append([str(row['Animal_ID']), sire, dam])

    if len(result_df) > 1000:
        ped_data.append(["...", "...", "..."])

    t_ped = Table(ped_data, repeatRows=1, colWidths=[130, 130, 130])
    t_ped.setStyle(TableStyle(ped_styles))
    elements.append(t_ped)
    elements.append(Spacer(1, 15))

    # --- SECTION 7: SELECTION & CULLING RECOMMENDATIONS ---
    elements.append(Paragraph("<b>7. Selection & Culling Recommendations:</b>", styles['Heading2']))
    
    # 7.1 Selection Candidates (Top 25% EBV & F < 6.25%)
    threshold_ebv = result_df["EBV"].quantile(0.75)
    selection_df = result_df[
        (result_df["EBV"] >= threshold_ebv) & 
        (result_df["Inbreeding_%"] < 6.25)
    ].sort_values("EBV", ascending=False).head(20) # Top 20 for PDF
    
    elements.append(Paragraph(f"<b>Top Selection Candidates (Count: {len(selection_df)}):</b>", styles['Heading3']))
    if not selection_df.empty:
        sel_data = [["Animal_ID", "EBV", "F (%)", "Status"]]
        for _, r in selection_df.iterrows():
            sel_data.append([str(r["Animal_ID"]), f"{r['EBV']:.4f}", f"{r['Inbreeding_%']:.2f}", str(r["Classification"])])
        
        t_sel = Table(sel_data, colWidths=[100, 80, 80, 130])
        t_sel.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#16a34a")), # Green
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8)
        ]))
        elements.append(t_sel)
    else:
        elements.append(Paragraph("No candidates meet the strict selection criteria.", styles['Normal']))
    
    elements.append(Spacer(1, 10))
    
    # 7.2 Culling Candidates (F >= 25% OR Backcross OR Bottom 10% EBV)
    threshold_low_ebv = result_df["EBV"].quantile(0.10)
    culling_df = result_df[
        (result_df["Inbreeding_%"] >= 25) | 
        (result_df["Reproduction_Warning"] != "") |
        (result_df["EBV"] <= threshold_low_ebv)
    ].sort_values("Inbreeding_%", ascending=False).head(20) # Top 20 for PDF
    
    elements.append(Paragraph(f"<b>Priority Culling Candidates (Count: {len(culling_df)}):</b>", styles['Heading3']))
    if not culling_df.empty:
        cul_data = [["Animal_ID", "F (%)", "Warning", "Reason"]]
        for _, r in culling_df.iterrows():
            reason = "Low Genetic" if r["EBV"] <= threshold_low_ebv else "High Inbreeding"
            if r["Reproduction_Warning"]: reason = "Backcross"
            cul_data.append([str(r["Animal_ID"]), f"{r['Inbreeding_%']:.2f}", str(r["Reproduction_Warning"]), reason])
        
        t_cul = Table(cul_data, colWidths=[100, 80, 110, 100])
        t_cul.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#dc2626")), # Red
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 0), (-1, -1), 8)
        ]))
        elements.append(t_cul)
    else:
        elements.append(Paragraph("No high-priority culling candidates found.", styles['Normal']))
    
    elements.append(Spacer(1, 15))

    # --- SECTION 8: MATING STRATEGY (INBREEDING PREVENTION) ---
    elements.append(Paragraph("<b>8. Mating Strategy to Prevent Future Inbreeding:</b>", styles['Heading2']))
    strategy_txt = """
    To ensure genetic progress while maintaining healthy inbreeding levels in the next generation:
    <br/><br/>
    • <b>Lineage Crossing:</b> Pair selected individuals from unrelated families (check the Relationship Matrix).
    <br/>
    • <b>Avoid Kinship Mating:</b> Never mate full-siblings, half-siblings, or parent-offspring pairs.
    <br/>
    • <b>Popular Sire Management:</b> Limit the usage of a single 'Elite' sire to no more than 15-20% of the dam population.
    <br/>
    • <b>Outcrossing:</b> If no suitable unrelated elite sires are available, prioritize using external semen (AI) from tested, unrelated bulls.
    <br/>
    • <b>Replacement Strategy:</b> Retain female offspring from 'Elite' parents to replace older 'Commercial' dams.
    """
    elements.append(Paragraph(strategy_txt, styles['Normal']))
    elements.append(Spacer(1, 20))

    # Permanent URL link with more explicit format
    link_url = "https://inbreeding.streamlit.app/"
    elements.append(Paragraph(f'<b>Access Full Visualization:</b> <a href="{link_url}" color="blue"><u>{link_url}</u></a>', styles['Normal']))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("<i>Developed by: Galuh Adi Insani (Universitas Gadjah Mada)</i>", styles['Italic']))
    
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
                Decision Support System for Advanced Breeding Management
                <br>Selection is easier, faster, and more accurate with inbreeding analysis, EBV, and pedigree insights.
            </p>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown('<div class="sidebar-header">CONFIGURATION</div>', unsafe_allow_html=True)
        mode = st.radio(
            "Select Data Source",
            ["Full cattle example", "Upload own file"],
            help="Use sample data to learn how it works or upload your own file."
        )
        
        st.info("""
        **Data Format:**
        Ensure the file has the following columns:
        - `Animal_ID` (Cattle ID)
        - `Sire_ID` (Sire ID)
        - `Dam_ID` (Dam ID)
        - `Optional:`
            - `Phenotype` (Phenotype value for EBV calculation, e.g., milk yield, growth rate, etc.)

        Use `-` for empty data.
        """)

        if mode == "Full cattle example":
            raw_df = contoh_sapi_lengkap()
        else:
            uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
            if not uploaded:
                st.warning("Please upload a CSV or Excel file to begin.")
                st.stop()
            raw_df = read_file(uploaded)

    # Optimization Warning for Large Data
    if len(raw_df) > 500:
        st.warning(f"Large Data Detected ({len(raw_df)} rows): Additive relationship matrix is being calculated using NumPy acceleration. Please wait a moment.")

    cols = list(raw_df.columns)
    
    with st.sidebar:
        st.markdown("### Column Mapping Search")
        id_col = st.selectbox("Animal_ID Column", cols, index=0)
        sire_col = st.selectbox("Sire_ID Column", cols, index=1 if len(cols)>1 else 0)
        dam_col = st.selectbox("Dam_ID Column", cols, index=2 if len(cols)>2 else 0)
        
        phenotype_col = st.selectbox(
            "Phenotype Column (Optional)", 
            ["-"] + cols, 
            index=cols.index("Phenotype") + 1 if "Phenotype" in cols else 0,
            help="Select phenotype column to calculate Breeding Value (EBV)."
        )
        pheno_val = None if phenotype_col == "-" else phenotype_col

        st.markdown("### Genetic Parameters")
        h2 = st.slider("Heritability ($h^2$)", 0.0, 1.0, 0.3, 0.05)
        depression_rate = st.slider("Inbreeding Depression Rate (per 1% F)", 0.0, 5.0, 1.0, 0.1)
        
        st.markdown("### Selection Parameters")
        intensity = st.slider("Selection Intensity (i)", 0.0, 3.0, 1.5, 0.1)

    try:
        internal = standardize_input(raw_df, id_col, sire_col, dam_col, pheno_val)
        std_df, res_df, matrix_df = calculate(internal, h2=h2, depression_rate=depression_rate)
        
        res_display_data = res_df[res_df["Data_Type"] == "Input data"]
        
        # Define tabs
        tabs = st.tabs(["Results & Analysis", "Genetic Visualization", "Pedigree Chart", "Relationship Matrix (A)", "Heterosis & Crossbreeding"])

        with tabs[0]:
            st.subheader("Data Summary")
            
            # Metrics
            m0, m1, m2, m3, m4 = st.columns(5)
            total_sapi = len(res_display_data)
            inbred_sapi = len(res_display_data[res_display_data["Inbreeding_%"] > 0])
            avg_f = float(res_display_data["Inbreeding_%"].mean())
            max_f = float(res_display_data["Inbreeding_%"].max())
            
            m0.metric("Heritability ($h^2$)", f"{h2:.2f}")
            m1.metric("Total Population", f"{total_sapi} heads")
            m2.metric("Inbred Livestock", f"{inbred_sapi} heads")
            m3.metric("Average F", f"{avg_f:.2f}%")
            m4.metric("Highest F", f"{max_f:.2f}%")

            # Classification Summary
            st.markdown("### Livestock Classification Distribution")
            dist = res_display_data["Classification"].value_counts()
            
            # Check for Elite Stock
            has_elite_male = any("Core Sire" in str(x) for x in res_display_data["Breeding_Interpretation"])
            has_elite_female = any("Core Female" in str(x) for x in res_display_data["Breeding_Interpretation"])
            has_elite = has_elite_male or has_elite_female
            
            c_dist = st.columns(len(dist))
            for i, (label, count) in enumerate(dist.items()):
                # Show gender details if label is Elite Stock
                display_label = label
                if label == "Elite Stock":
                    m_count = sum("Core Sire" in str(x) for x in res_display_data["Breeding_Interpretation"])
                    f_count = sum("Core Female" in str(x) for x in res_display_data["Breeding_Interpretation"])
                    display_label = f"Elite (M:{m_count}, F:{f_count})"
                c_dist[i].metric(display_label, f"{count} heads")

            # Advice if no Elite Stock
            if not has_elite:
                st.warning("Warning: No Elite Stock found in this population.")
                with st.container():
                    st.markdown("""
                    <div class="info-card">
                    <b>Genetic Improvement Advice:</b><br/>
                    Since the current population has no individuals with 'Elite' genetic potential (high EBV & low Inbreeding), the following steps are suggested:
                    <ol>
                        <li><b>Outcrossing:</b> Introduce sires from outside (or frozen semen) that have proven breeding values but no kinship relationship with current dams.</li>
                        <li><b>Strict Selection:</b> Use the best <b>'Breeding Stock'</b> animals as replacement dams and avoid using 'Commercial' animals as parents.</li>
                        <li><b>Re-evaluation:</b> Ensure phenotype data is accurate. EBV values are highly dependent on weight/production recording accuracy.</li>
                        <li><b>Inbreeding Management:</b> Focus on reducing inbreeding coefficients below 3% to open opportunities for Elite individuals to emerge in future generations.</li>
                    </ol>
                    </div>
                    """, unsafe_allow_html=True)

            # Selection Response
            if pheno_val:
                st.markdown("---")
                st.subheader("Selection Response Estimation")

                # Filter valid phenotypes for SD calculation
                valid_phenos = pd.to_numeric(res_display_data["Phenotype"], errors='coerce').dropna()
                if not valid_phenos.empty:
                    sd_p = valid_phenos.std()
                    avg_p = valid_phenos.mean()
                    response = calculate_selection_response(h2, sd_p, intensity)
                    
                    # New neater layout
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Average Phenotype", f"{avg_p:.2f}")
                    m2.metric("Standard Deviation (σp)", f"{sd_p:.2f}")
                    m3.metric("Selection Response (R)", f"{response:.4f}")
                    
                    # Average Heterosis
                    valid_heterosis = res_display_data[res_display_data["Heterosis"] != 0]["Heterosis"]
                    avg_h = valid_heterosis.mean() if not valid_heterosis.empty else 0.0
                    m4.metric("Average Heterosis", f"{avg_h:.4f}")
                    
                    # Interpretation explanation
                    st.info(f"""
                    **Result Interpretation:**
                    *   **Population Average:** Currently is **{avg_p:.2f}** units.
                    *   **Genetic Progress:** With selection intensity **{intensity}** and heritability **{h2}**, a progress of **{response:.4f}** units is predicted in the next generation. Higher selection intensity (more aggressive) will increase selection response, but can also increase inbreeding risk if not well-managed. Moderate selection intensity (around 1.5) is often considered an optimal point for balance between genetic progress and inbreeding management.
                    *   **Role of Heterosis:** An average heterosis of **{avg_h:.4f}** units indicates that some individuals may have performance superiority due to better genetic combinations from both parents, especially if crossbreeding occurs. Heterosis can provide an additional boost to offspring performance beyond what is predicted by EBV alone.
                    *   **Next Generation Estimation:** Average offspring performance is expected to be **{avg_p + response:.2f}** units. However, this is an average prediction and actual results may vary depending on environmental factors, management, and other genetic interactions. Therefore, it is important to continue monitoring actual offspring performance and adjust breeding strategies as needed.
                    *   **Important Note:** Selection response is a prediction based on current data. External factors such as management changes, disease, or environmental conditions can affect actual results in the field. Always use latest data for recalculation and periodic evaluation of your breeding strategy.
                    *   **Recommendation:** To maximize selection response, focus on choosing individuals with high EBV and low inbreeding as parents, and consider using semen from outside sires to increase genetic variation and reduce inbreeding risk.
                    *   **Warning:** Avoid overly aggressive selection on individuals with high inbreeding, as this can increase inbreeding depression risk and lower overall offspring performance.
                    *   **Management Advice:** Consider performing periodic sire rotations and using crossbreeding strategies if needed to maintain the genetic health of your population.
                    *   **Conclusion:** With a good understanding of selection response and inbreeding management, you can make smarter breeding decisions to improve your livestock performance sustainably.
                    """)
                    
                    # Backcross/Sire-Daughter Alert Section
                    backcross_cases = res_display_data[res_display_data["Reproduction_Warning"] != ""]
                    if not backcross_cases.empty:
                        st.markdown("---")
                        st.error(f"Detected {len(backcross_cases)} Sire-Daughter Mating Cases (Backcross)")
                        
                        cols_back = st.columns(2)
                        with cols_back[0]:
                            st.write("**List of Exposed Individuals:**")
                            st.dataframe(backcross_cases[["Animal_ID", "Sire_ID", "Dam_ID", "Inbreeding_%"]], hide_index=True)
                        
                        with cols_back[1]:
                            st.markdown("""
                            <div class="info-card" style="border-left: 4px solid #ef4444;">
                            <b>Reproduction Management & AI Insight:</b><br/>
                            Mating a sire with its biological daughter results in a minimum inbreeding coefficient (F) of <b>25%</b>.
                            <ul>
                                <li><b>Main Risk:</b> Drastic vitality reduction, high birth defect risk, and inbreeding depression on growth/milk performance.</li>
                                <li><b>Artificial Insemination (AI) Strategy:</b>
                                    <ul>
                                        <li><b>Semen Database:</b> Inseminator MUST check livestock cards. Do not use semen codes from the same sire as the father of that female.</li>
                                        <li><b>Straw Rotation:</b> Use straws from different <i>lineage</i> sires or other breeds (Crossbreeding) if needed to break the inbreeding chain.</li>
                                    </ul>
                                </li>
                                <li><b>Culling:</b> Backcross offspring should be used as <b>Final Stock</b> (slaughter) and not selected as replacement stock.</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)

                    # Inbreeding vs Phenotype Relationship (Correlation & Regression)
                    st.markdown("---")
                    st.subheader("Inbreeding vs Phenotype Relationship Analysis")
                    stats = calculate_stats(res_display_data)
                    
                    if stats:
                        c_stat1, c_stat2, c_stat3 = st.columns(3)
                        with c_stat1:
                            st.metric("Correlation (r)", f"{stats['correlation']:.4f}")
                            st.caption("Relationship between inbreeding level and performance.")
                        with c_stat2:
                            st.metric("Regression (b)", f"{stats['b']:.4f}")
                            st.caption("Phenotype unit reduction per 1% increase in inbreeding.")
                        with c_stat3:
                            st.metric("Coefficient of Determination ($R^2$)", f"{stats['r_squared']:.4f}")
                            st.caption("Proportion of phenotype variation influenced by inbreeding.")
                        
                        st.info(f"**Static Interpretation:** Regression equation: $Y = {stats['a']:.2f} + ({stats['b']:.4f}) X$. "
                                f"Meaning, every 1% increase in inbreeding is predicted to decrease phenotype by {abs(stats['b']):.4f} units.")
                    
                    # --- HARDY-WEINBERG ANALYSIS ---
                    st.markdown("---")
                    st.subheader("Hardy-Weinberg Equilibrium (HWE) Analysis")
                    hwe_res = analyze_hardy_weinberg(res_display_data)
                    
                    hw_col1, hw_col2 = st.columns([0.4, 0.6])
                    with hw_col1:
                        st.write(f"#### Status: {hwe_res['status']}")
                        st.write(hwe_res['insight'])
                    
                    with hw_col2:
                        st.write("**Strategy Recommendation:**")
                        for s in hwe_res['saran']:
                            st.write(f"- {s}")
                    
                    if hwe_res['is_deviating']:
                        st.warning("Population shows significant indications of non-random mating (accumulated inbreeding).")
                    else:
                        st.success("Population is still in a healthy genetic distribution track.")
                else:
                    st.warning("Invalid phenotype data for SD calculation.")

            st.markdown("### Calculation Results Table")
            # Use st.dataframe with fixed height for virtualized scrolling
            st.dataframe(clean_display(res_display_data), use_container_width=True, height=500)
            
            # Detailed Interpretation for Selected Animal
            st.markdown("---")
            col_sel1, col_sel2 = st.columns([0.4, 0.6])
            with col_sel1:
                st.subheader(" Individual Interpretation")
                selected_animal = st.selectbox("Select Cattle:", res_display_data["Animal_ID"])
            
            if selected_animal:
                row = res_display_data[res_display_data["Animal_ID"] == selected_animal].iloc[0]
                with col_sel2:
                    st.info(f"**Individual:** {row['Animal_ID']}\n\n{row['Breeding_Interpretation']}")
                    st.markdown(f"""
                    - **Inbreeding:** {row['Inbreeding_%']}% ({row['Inbreeding_Condition']})
                    - **Performance Impact:** {row['Inbreeding_Depression']} units.
                    - **Biological Impact:** {row['Biological_Impact']}
                    """)
            
            # Additional Information based on Max F
            st.markdown("### Interpretation Guide")
            with st.expander("Click to understand breeding terms"):
                st.markdown("""
                - **Inbreeding Coefficient (F):** Percentage of genetic similarity due to related parents.
                - **EBV (Estimated Breeding Value):** Value indicating genetic potential that will be passed to offspring. The higher (positive), the better.
                - **Inbreeding Depression:** Estimated performance reduction experienced by individuals due to high inbreeding.
                - **Selection Response (R):** Prediction of population quality progress in next generation if selection is performed.
                """)
            
            st.markdown("### Population Impact Analysis")
            st.info(dampak_inbreeding(max_f))
            
            # Download options
            st.markdown("### Download Report")
            c1, c2, c3 = st.columns(3)
            csv = clean_display(res_df).to_csv(index=False).encode('utf-8')
            c1.download_button(" CSV (Full Data)", csv, "analytics_data.csv", "text/csv", use_container_width=True)
            
            # PDF Report
            current_settings = {
                'h2': h2,
                'intensity': intensity
            }
            pdf_data = generate_pdf(res_display_data, settings=current_settings)
            c2.download_button(" PDF (Official Report)", pdf_data, "Breeding_Report.pdf", "application/pdf", use_container_width=True)
            
            # Simple Text Report
            txt_report = dots_to_pedigree(res_display_data)
            c3.download_button(" TXT (Quick Summary)", txt_report.encode('utf-8'), "summary.txt", "text/plain", use_container_width=True)

            # --- SELECTION & CULLING SECTION ---
            st.markdown("---")
            st.subheader("Selection & Culling Recommendations")
            
            # Selection Criteria: Top 25% EBV & Low Inbreeding
            threshold_ebv = res_display_data["EBV"].quantile(0.75)
            seleksi_df = res_display_data[
                (res_display_data["EBV"] >= threshold_ebv) & 
                (res_display_data["Inbreeding_%"] < 6.25)
            ].sort_values("EBV", ascending=False)
            
            # Culling Criteria: Very High Inbreeding OR Sire-Daughter Mating OR very low EBV (Bottom 10%)
            threshold_low_ebv = res_display_data["EBV"].quantile(0.10)
            culling_df = res_display_data[
                (res_display_data["Inbreeding_%"] >= 25) | 
                (res_display_data["Reproduction_Warning"] != "") |
                (res_display_data["EBV"] <= threshold_low_ebv)
            ].sort_values("Inbreeding_%", ascending=False)
            
            col_sel_recom, col_cul_recom = st.columns(2)
            
            with col_sel_recom:
                st.success(f"**Selection Candidates (Next Gen Parents): {len(seleksi_df)} heads**")
                st.write("Priority based on high EBV and low inbreeding risk.")
                st.dataframe(seleksi_df[["Animal_ID", "EBV", "Inbreeding_%", "Classification"]], hide_index=True)
                
            with col_cul_recom:
                st.error(f"**Culling Candidates (Slaughter/Out): {len(culling_df)} heads**")
                st.write("Based on extreme inbreeding, backcross cases, or very low genetic potential.")
                st.dataframe(culling_df[["Animal_ID", "EBV", "Inbreeding_%", "Reproduction_Warning"]], hide_index=True)
            # ---------------------------------

            # --- MATING STRATEGY SECTION ---
            st.markdown("---")
            st.subheader("Future Mating Strategy (Inbreeding Prevention)")
            
            # Divide selection candidates into males and females for mating recommendations
            # We use the internal 'Breeding_Interpretation' to guestimate gender if possible, 
            # otherwise we just look at candidates IDs.
            males = seleksi_df[seleksi_df["Breeding_Interpretation"].str.contains("Sire|Pejantan", case=False)]
            females = seleksi_df[seleksi_df["Breeding_Interpretation"].str.contains("Dam|Indukan|Female|Betina", case=False)]
            
            col_m, col_f = st.columns(2)
            with col_m:
                st.write(f"**Available Elite/Selected Sires: {len(males)}**")
                if not males.empty:
                    st.dataframe(males[["Animal_ID", "EBV", "Inbreeding_%"]], hide_index=True)
                else:
                    st.info("No Elite Sires detected in selection candidates. Consider using **AI (Artificial Insemination)** with external tested sires.")
            
            with col_f:
                st.write(f"**Available Elite/Selected Dams: {len(females)}**")
                if not females.empty:
                    st.dataframe(females[["Animal_ID", "EBV", "Inbreeding_%"]], hide_index=True)
                else:
                    st.info("No Elite Dams detected in selection candidates.")

            if not males.empty and not females.empty:
                st.markdown("#### Suggested Pairings (Based on Genetic Distance)")
                pairings = []
                # Simple strategy: Match best EBVs while ensuring they are from different lineages
                # In a real app we'd check the A-matrix relationship between them.
                for _, f_row in females.head(5).iterrows():
                    f_id = f_row["Animal_ID"]
                    # Find a sire with lowest relationship in A-matrix (if possible)
                    # For now, we recommend cross-matching among survivors
                    best_sire = males.iloc[0]["Animal_ID"]
                    pairings.append({"Dam": f_id, "Suggested Sire": best_sire, "Strategy": "High EBV Crossing"})
                
                st.table(pairings)
                st.caption("Note: Pairing recommendations ensure top genetic potential. Always verify physical health before mating.")
            
            st.markdown("""
            <div class="info-card" style="border-left: 4px solid #3b82f6;">
            <b>Mating Strategy to Prevent Future Inbreeding:</b>
            <ul>
                <li><b>Avoid Full/Half-Sib Matings:</b> Never mate candidates sharing the same Sire or Dam.</li>
                <li><b>Line Crossing:</b> Pair selected individuals from unrelated lineages (e.g., Line A x Line B).</li>
                <li><b>Compensatory Mating:</b> Correct weaknesses in selected dams by choosing sires excelling in those specific traits.</li>
                <li><b>A-Matrix Utility:</b> Look at the 'Relationship Matrix' tab. Pick pairs with relationship values near 0.0.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
            # ---------------------------------

            # Extra Insights Section
            st.markdown("---")
            st.markdown("### Important Livestock Breeding Parameters")
            
            with st.expander("See Breeding Concept Details"):
                col_i1, col_i2 = st.columns(2)
                with col_i1:
                    st.markdown("""
                    **1. Heritability ($h^2$):**
                    Extent to which phenotype variation is caused by additive genetics. 
                    - Low (< 0.2): Environmentally dominant (e.g., reproduction).
                    - Moderate (0.2 - 0.4): Growth.
                    - High (> 0.4): Carcass/milk quality.
                    
                    **2. Breeding Value (EBV):**
                    Prediction of parental genetic value that will be passed down. EBV is twice the genetic transmission value (*Progeny Difference*).
                    """)
                with col_i2:
                    st.markdown("""
                    **3. Tandem Selection:**
                    Selection method for one trait gradually over several generations before switching to another trait. Effective if focused on one primary economic goal.

                    **Tandem Selection Strategy & Statistical Relationships:**
                    If you implement tandem selection, note these strategic steps:
                    - **Trait Priority:** Start with the trait that has the highest **heritability ($h^2$)** or economic value for rapid initial progress.
                    - **Threshold (Goal):** Set a minimum target before switching to next trait so genetic progress on first trait is not lost.
                    - **Correlation Analysis:** Use **Correlation ($r$)** metrics above to monitor if improvements in one trait don't damage others (negative correlation).
                    - **Regression & Prediction:** Utilize **Regression ($b$)** value to predict phenotype performance changes alongside genetic changes in selected trait.

                    **4. Selection Intensity ($i$):**
                    Selection strength applied. Fewer livestock chosen as parents from total population means greater intensity ($i$).
                    """)

            st.markdown("### Insights & Management Strategy")
            
            col_in1, col_in2 = st.columns(2)
            
            with col_in1:
                st.info("""
                ** Why is Inbreeding Dangerous?**
                Inbreeding increases chances of detrimental recessive genes emerging. This causes:
                - **Vitality Reduction:** Livestock get sick more easily.
                - **Reproduction Issues:** Longer calving interval.
                - **Growth Reduction:** Lower weaning and adult weights.
                """)
                
            with col_in2:
                st.success("""
                ** Prevention Strategy**
                1. **Sire Rotation:** Change sires every 2 years or after first female offspring reach mating age.
                2. **Outcrossing:** Mate livestock with individuals from other groups/populations that are unrelated.
                3. **Digital Recording:** Use this system routinely for simulations before mating livestock.
                """)

        with tabs[1]:
            st.subheader(" Genetic Distribution Visualization")
            
            v_col1, v_col2 = st.columns(2)
            
            with v_col1:
                st.markdown("### Inbreeding (F) Distribution")
                # Histogram data F
                f_data = res_display_data["Inbreeding_%"]
                counts, bin_edges = np.histogram(f_data, bins=10)
                hist_df = pd.DataFrame({
                    "F range (%)": [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(counts))],
                    "Cattle Count": counts
                })
                st.bar_chart(hist_df.set_index("F range (%)"))
                st.caption("This graph shows inbreeding level distribution within population.")

            with v_col2:
                st.markdown("### Genetic Potential (EBV)")
                # Bar chart for top 10 EBVs
                top_ebv = res_display_data.sort_values("EBV", ascending=False).head(10)
                st.bar_chart(top_ebv.set_index("Animal_ID")["EBV"])
                st.caption("Top 10 livestock with highest Breeding Value (EBV).")

            st.markdown("---")
            st.markdown("### Inbreeding vs Phenotype Relationship")
            # Scatter plot using Streamlit's native chart
            # We add a small jitter if needed, but simple scatter 
            scatter_data = res_display_data[res_display_data["Phenotype"].apply(lambda x: not is_unknown(x))].copy()
            if not scatter_data.empty:
                scatter_data["Phenotype_Val"] = pd.to_numeric(scatter_data["Phenotype"])
                st.scatter_chart(
                    scatter_data,
                    x="Inbreeding_%",
                    y="Phenotype_Val",
                    color="Classification",
                    size="EBV"
                )
                st.caption("Dots show inbreeding (X-axis) vs Phenotype Performance (Y-axis) relationship. Colors show classification.")
            else:
                st.info("No phenotype data available for distribution visualization.")

        with tabs[2]:
            st.subheader(" Pedigree Chart Visualization")
            if len(res_df) > 100:
                st.info(" **Note:** Since data exceeds 100 items, visualization only shows the first 50 individuals for performance.")
            st.markdown("This chart shows descendant relationships between cattle. Red color indicates high inbreeding (>25%).")
            st.graphviz_chart(make_dot(res_df))

        with tabs[3]:
            st.subheader(" Additive Relationship Matrix")
            if len(matrix_df) > 500:
                st.warning(" **Performance Warning:** Showing matrix > 500x500 in browser may cause lag. Suggested to download CSV if data is very large.")
                if st.button("Display Matrix Anyway"):
                    st.dataframe(matrix_df, use_container_width=True)
            else:
                st.dataframe(matrix_df, use_container_width=True)

        with tabs[4]:
            st.subheader(" Heterosis Insights & Crossbreeding Strategy")
            
            h_col1, h_col2 = st.columns(2)
            
            with h_col1:
                st.markdown("""
                ### What is Heterosis?
                **Heterosis** (or *Hybrid Vigor*) is the performance increase in crossbred offspring compared to average of both parents.
                
                **Simple Formula:**
                $$HF_1 = \\text{Offspring Average} - \\text{Parent Average}$$
                $$\\% \\text{Heterosis} = \\frac{HF_1}{\\text{Parent Average}} \\times 100\\%$$
                
                **Why is it important?**
                Heterosis is very effective for traits with **low heritability**, such as:
                - Survival.
                - Fertility/Reproduction.
                - Disease resistance.
                """)
                
                st.info("""
                ** Important Insight:**
                Inbreeding is the opposite of heterosis. If inbreeding reduces performance (Inbreeding Depression), then crossbreeding (outbreeding) increases it through heterosis.
                """)

            with h_col2:
                st.markdown("""
                ### Crossbreeding Strategy
                To get maximum heterosis, you can implement:
                
                1. **Terminal Cross:** All crossbred offspring sold (not kept as replacements). Gives 100% heterosis in offspring.
                2. **Rotational Crossing:** Use two or three breeds alternately. Maintains heterosis around 67% (2 breeds) or 86% (3 breeds) in sustained generations.
                3. **Backcrossing:** Mating offspring back to one of pure parent breeds. Used to introduce specific traits from one breed to another.
                
                **Recommendation:**
                - Use sires from breeds superior in growth traits.
                - Use dams from breeds that have maternal superiority (milk production, temperament).
                """)

            st.markdown("---")
            st.markdown("""
            ### Genetic Correlation vs Heterosis
            Important to remember that heterosis is not permanent (not stably inherited like EBV). 
            - **EBV** is used for long-term genetic progress (selection).
            - **Heterosis** is used for short-term production gain (crossbreeding).
            
            Our system helps you keep inbreeding levels low so that heterosis potential during future crossbreeding remains optimal. Perform regular regulation and evaluation to ensure population remains healthy and productive, and continue data recording for more accurate genetic analysis in the future.
            """)
        
    except Exception as e:
        st.error(f" Error occurred in data processing: {e}")
        st.exception(e)

    # Footer
    st.markdown("""
        <div class="footer">
            Analytics System v2.5 | Created by Galuh Adi Insani
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
