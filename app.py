import io
import pathlib
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False


# ============================================================
# BREEDING & INBREEDING ANALYTICS - APP.PY
# Informative English Version
# ============================================================

EMPTY = "-"

UNKNOWN_VALUES = {
    "", " ", "-", "--", "0", "na", "n/a", "nan", "none", "null",
    "unknown", "empty",
}


def is_unknown(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    text = str(value).strip()
    return text.lower() in UNKNOWN_VALUES


def clean_id(value) -> Optional[str]:
    if is_unknown(value):
        return None

    text = str(value).strip()

    # Fix numeric IDs from Excel, for example 101.0 becomes 101
    if text.endswith(".0"):
        try:
            num = float(text)
            if num.is_integer():
                text = str(int(num))
        except Exception:
            pass

    return text


def show_value(value) -> str:
    if is_unknown(value):
        return EMPTY
    return str(value).strip()


def clean_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in out.columns:
        if pd.api.types.is_numeric_dtype(out[col]):
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        else:
            out[col] = out[col].astype(str).apply(
                lambda x: EMPTY if is_unknown(x) else x
            )

    return out


def kondisi_inbreeding(percent: float) -> str:
    if percent <= 0:
        return "Not inbred"
    if percent < 6.25:
        return "Low inbreeding"
    if percent < 12.5:
        return "Moderate inbreeding"
    if percent < 25:
        return "High inbreeding"
    return "Very high inbreeding"


def rekomendasi(percent: float) -> str:
    if percent <= 0:
        return "Safe based on pedigree: no inbreeding detected."
    if percent < 6.25:
        return "Low risk, but pedigree monitoring is still recommended."
    if percent < 12.5:
        return "Requires attention. Avoid repeated mating between related animals."
    if percent < 25:
        return "High risk. Use mates from unrelated lineages."
    return "Very high risk. Avoid close-relative mating and do not prioritize as breeding stock."


def dampak_inbreeding(percent: float) -> str:
    if percent <= 0:
        return "Impact: Genetic variation is maintained. No inbreeding depression risk was detected from the available pedigree."
    if percent < 6.25:
        return "Impact: Minimal short-term influence, but cumulative effects may appear if related mating continues across generations."
    if percent < 12.5:
        return "Impact: Possible reduction in production performance, fertility, growth, milk yield, and higher risk of recessive genetic problems."
    if percent < 25:
        return "Impact: Inbreeding depression may become visible through weaker immunity, lower fertility, slow growth, and possible birth defects."
    return "Critical impact: Very high risk of embryo loss, infertility, low vitality, congenital defects, and long-term decline in genetic quality."


def contoh_sapi_lengkap() -> pd.DataFrame:
    return pd.DataFrame({
        "Animal_ID": [
            "SIRE_01", "DAM_01", "COW_C", "COW_D", "COW_X",
            "COW_B", "COW_A", "COW_E", "COW_F"
        ],
        "Sire_ID": [
            "-", "-", "SIRE_01", "SIRE_01", "SIRE_01",
            "COW_D", "COW_B", "COW_B", "COW_D"
        ],
        "Dam_ID": [
            "-", "-", "DAM_01", "DAM_01", "DAM_01",
            "COW_C", "COW_C", "-", "-"
        ],
        "Phenotype": [550, 420, 480, 500, 490, 460, 470, 430, 440],
    })


def calculate_selection_response(h2: float, sd_p: float, intensity: float) -> float:
    """
    Selection response formula:
    R = i * h2 * sigma_p
    """
    return intensity * h2 * sd_p


def calculate_inbreeding_depression(f_percent: float, depression_rate: float) -> float:
    """
    Estimated performance reduction:
    Depression = F (%) * depression rate per 1% F
    """
    return f_percent * depression_rate


def analyze_hardy_weinberg(result_df: pd.DataFrame) -> Dict:
    avg_f = result_df["Inbreeding_%"].mean() / 100.0
    is_deviating = avg_f > 0.05

    status = "Deviation Occurring - Not in Equilibrium" if is_deviating else "Near Equilibrium"

    insight = (
        f"The average population inbreeding coefficient is {avg_f:.4f}. "
        "In population genetics, this value can be used as an indicator of deviation "
        "from the ideal random mating condition."
    )

    if is_deviating:
        suggestions = [
            "Outcrossing: introduce unrelated sires or semen from outside the population.",
            "Increase effective population size by using more active sires.",
            "Apply sire rotation and avoid excessive use of one popular sire.",
            "Consider crossbreeding for commercial production to restore heterozygosity.",
        ]
    else:
        suggestions = [
            "The current mating pattern still appears able to maintain genetic variation.",
            "Continue monitoring pedigrees to prevent sudden increases in inbreeding.",
        ]

    return {
        "status": status,
        "insight": insight,
        "saran": suggestions,
        "is_deviating": is_deviating,
    }


def standardize_input(raw_df, id_col, sire_col, dam_col, phenotype_col=None):
    cols = [id_col, sire_col, dam_col]

    if phenotype_col and phenotype_col in raw_df.columns:
        cols.append(phenotype_col)
        df = raw_df[cols].copy()
        df.columns = ["Animal_ID", "Sire_ID", "Dam_ID", "Phenotype"]
    else:
        df = raw_df[cols].copy()
        df.columns = ["Animal_ID", "Sire_ID", "Dam_ID"]

    for col in df.columns:
        if col != "Phenotype":
            df[col] = df[col].apply(clean_id)

    df = df.dropna(subset=["Animal_ID"]).copy()

    if df.empty:
        raise ValueError("No valid Animal_ID was found.")

    return df.reset_index(drop=True)


def klasifikasi_ternak(ebv: float, f_percent: float) -> str:
    if ebv > 0.5 and f_percent < 3.125:
        return "Elite Stock"
    if ebv > 0 and f_percent < 6.25:
        return "Breeding Stock"
    if ebv > 1.0:
        return "Line Breeding"
    if f_percent > 12.5:
        return "Final Stock - Slaughter Only"
    return "Commercial"


def interpretasi_pemuliaan(
    ebv: float,
    f_percent: float,
    animal_id: str,
    results_df: Optional[pd.DataFrame] = None,
) -> str:
    status_ebv = "Superior" if ebv > 0 else "Below average"
    classification = klasifikasi_ternak(ebv, f_percent)

    is_dam = False
    if results_df is not None and "Dam_ID" in results_df.columns:
        is_dam = animal_id in results_df["Dam_ID"].astype(str).values

    msg = f"Genetic Status: {status_ebv} with EBV {ebv:.2f}. "

    if classification == "Elite Stock":
        if is_dam:
            msg += (
                "Classification: Elite Stock - Core Female. "
                "Recommendation: highly suitable as a core dam or embryo donor candidate."
            )
        else:
            msg += (
                "Classification: Elite Stock - Core Sire. "
                "Recommendation: highly suitable as a main sire candidate or frozen semen source."
            )
    elif classification == "Breeding Stock":
        msg += (
            "Classification: Breeding Stock. "
            "Recommendation: suitable as replacement breeding stock for the next generation."
        )
    elif classification == "Line Breeding":
        msg += (
            "Classification: Line Breeding. "
            "Recommendation: strong genetic potential, but kinship must be controlled carefully."
        )
    elif classification == "Final Stock - Slaughter Only":
        msg += (
            "Classification: Final Stock - Slaughter Only. "
            "Recommendation: not recommended for breeding because of high inbreeding risk."
        )
    else:
        msg += (
            "Classification: Commercial. "
            "Recommendation: suitable for commercial production, but not prioritized for genetic improvement."
        )

    return msg


def calculate_stats(df: pd.DataFrame):
    try:
        if "Phenotype" not in df.columns:
            return None

        valid_df = df[df["Phenotype"].apply(lambda x: not is_unknown(x))].copy()

        if len(valid_df) < 3:
            return None

        valid_df["Phenotype"] = pd.to_numeric(valid_df["Phenotype"], errors="coerce")
        valid_df["F"] = pd.to_numeric(valid_df["Inbreeding_%"], errors="coerce")
        valid_df = valid_df.dropna(subset=["Phenotype", "F"])

        if len(valid_df) < 3:
            return None

        correlation = valid_df["F"].corr(valid_df["Phenotype"])

        x = valid_df["F"]
        y = valid_df["Phenotype"]
        n = len(valid_df)

        denominator = (n * (x ** 2).sum() - (x.sum()) ** 2)

        if denominator == 0:
            return None

        b = (n * (x * y).sum() - x.sum() * y.sum()) / denominator
        a = (y.sum() - b * x.sum()) / n
        r_squared = correlation ** 2 if correlation == correlation else 0

        return {
            "correlation": correlation,
            "b": b,
            "a": a,
            "r_squared": r_squared,
        }
    except Exception:
        return None


def calculate(df_input, h2=0.3, depression_rate=1.0):
    animal_ids = set(df_input["Animal_ID"])
    parent_ids = set(df_input["Sire_ID"].dropna()).union(set(df_input["Dam_ID"].dropna()))
    missing = sorted(parent_ids - animal_ids)

    founder_df = pd.DataFrame({
        "Animal_ID": missing,
        "Sire_ID": [None] * len(missing),
        "Dam_ID": [None] * len(missing),
    })

    if "Phenotype" in df_input.columns:
        founder_df["Phenotype"] = None

    df_full = pd.concat([founder_df, df_input], ignore_index=True)

    parents_map = {}
    pheno_map = {}

    for row in df_full.itertuples(index=False):
        if row.Animal_ID:
            s_val = None if is_unknown(row.Sire_ID) else str(row.Sire_ID)
            d_val = None if is_unknown(row.Dam_ID) else str(row.Dam_ID)

            parents_map[str(row.Animal_ID)] = (s_val, d_val)

            if hasattr(row, "Phenotype"):
                pheno_map[str(row.Animal_ID)] = row.Phenotype

    order = []
    state = {}

    def visit(animal):
        if animal is None:
            return

        status = state.get(animal, 0)

        if status == 1:
            raise ValueError(f"Pedigree cycle detected at {animal}. Please check the parent records.")

        if status == 2:
            return

        state[animal] = 1
        sire, dam = parents_map.get(animal, (None, None))

        for parent in [sire, dam]:
            if parent is not None:
                visit(parent)

        state[animal] = 2
        order.append(animal)

    for animal in list(parents_map.keys()):
        visit(animal)

    n = len(order)
    idx_map = {animal: i for i, animal in enumerate(order)}
    A = np.zeros((n, n), dtype=np.float32)

    phenotype_values = [
        pd.to_numeric(v, errors="coerce")
        for v in pheno_map.values()
        if v is not None and not is_unknown(v)
    ]
    phenotype_values = [v for v in phenotype_values if not pd.isna(v)]
    pop_avg = np.mean(phenotype_values) if phenotype_values else 0

    parent_indices = []
    for animal in order:
        sire, dam = parents_map.get(animal, (None, None))
        si = idx_map.get(sire) if sire is not None else None
        di = idx_map.get(dam) if dam is not None else None
        parent_indices.append((si, di))

    rows = []

    for i in range(n):
        si, di = parent_indices[i]

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

        animal_id = order[i]
        p_val = pheno_map.get(animal_id)

        ebv = 0.0
        if p_val is not None and not is_unknown(p_val):
            try:
                ebv = h2 * (float(p_val) - pop_avg)
            except Exception:
                ebv = 0.0

        heterosis = 0.0
        if si is not None and di is not None and p_val is not None and not is_unknown(p_val):
            p_sire = pheno_map.get(order[si])
            p_dam = pheno_map.get(order[di])

            if not is_unknown(p_sire) and not is_unknown(p_dam):
                try:
                    p_anak = float(p_val)
                    p_avg_parents = 0.5 * (float(p_sire) + float(p_dam))
                    heterosis = p_anak - p_avg_parents
                except Exception:
                    heterosis = 0.0

        depression = calculate_inbreeding_depression(F * 100, depression_rate)

        rows.append({
            "Animal_ID": animal_id,
            "Sire_ID": parents_map.get(animal_id)[0],
            "Dam_ID": parents_map.get(animal_id)[1],
            "Phenotype": show_value(p_val),
            "EBV": round(float(ebv), 4),
            "Heterosis": round(float(heterosis), 4),
            "Inbreeding_Depression": round(float(depression), 4),
            "Inbreeding_%": round(float(F * 100), 4),
            "Data_Type": "Additional founder" if animal_id in missing else "Input data",
        })

    res_df = pd.DataFrame(rows)

    res_df["Classification"] = res_df.apply(
        lambda r: klasifikasi_ternak(r["EBV"], r["Inbreeding_%"]),
        axis=1,
    )

    res_df["Breeding_Interpretation"] = res_df.apply(
        lambda r: interpretasi_pemuliaan(
            r["EBV"],
            r["Inbreeding_%"],
            r["Animal_ID"],
            res_df,
        ),
        axis=1,
    )

    res_df["Inbreeding_Condition"] = res_df["Inbreeding_%"].apply(kondisi_inbreeding)
    res_df["Biological_Impact"] = res_df["Inbreeding_%"].apply(dampak_inbreeding)
    res_df["Recommendation"] = res_df["Inbreeding_%"].apply(rekomendasi)

    res_df["Reproduction_Warning"] = ""

    for idx, row in res_df.iterrows():
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]

        if sire and dam:
            grand_sire_of_dam, _ = parents_map.get(str(dam), (None, None))
            if str(sire) == str(grand_sire_of_dam):
                res_df.at[idx, "Reproduction_Warning"] = "SIRE-DAUGHTER MATING DETECTED"

    matrix_df = pd.DataFrame(A, index=order, columns=order)

    return clean_display(df_input), res_df, matrix_df


def read_file(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file, dtype=str, keep_default_na=False)
    return pd.read_excel(uploaded_file, dtype=str, keep_default_na=False)


def dot_escape(value):
    return str(value).replace("\\", "\\\\").replace('"', '\\"')


def make_dot(result_df, max_nodes=50):
    df = result_df.head(max_nodes)
    animal_set = set(df["Animal_ID"].astype(str))

    dot = [
        "digraph Pedigree {",
        "rankdir=LR;",
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10];',
        "edge [arrowsize=0.6];",
    ]

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        f_value = float(row["Inbreeding_%"])
        classification = row.get("Classification", "")

        fill = "#FFFFFF"
        border_color = "black"
        penwidth = "1.0"

        if f_value >= 25:
            fill = "#FFE4E1"
        elif f_value > 0:
            fill = "#FFF4CC"

        if classification == "Elite Stock":
            border_color = "#FFD700"
            penwidth = "3.0"
        elif classification == "Breeding Stock":
            border_color = "#32CD32"
            penwidth = "2.0"
        elif "Final Stock" in classification:
            border_color = "#DC2626"
            penwidth = "2.0"

        label = f"{animal}\\nF={f_value:.2f}%\\n{classification}"

        dot.append(
            f'"{animal}" [label="{label}", fillcolor="{fill}", color="{border_color}", penwidth="{penwidth}"];'
        )

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]

        if sire and str(sire) in animal_set:
            dot.append(f'"{dot_escape(sire)}" -> "{animal}" [label="sire"];')

        if dam and str(dam) in animal_set:
            dot.append(f'"{dot_escape(dam)}" -> "{animal}" [label="dam"];')

    dot.append("}")
    return "\n".join(dot)


def dots_to_pedigree(result_df, settings=None):
    lines = [
        "LIVESTOCK BREEDING & INBREEDING SUMMARY",
        "=" * 60,
        f"Printed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
    ]

    avg_f = result_df["Inbreeding_%"].mean()
    avg_ebv = result_df["EBV"].mean()

    lines.append("1. POPULATION STATISTICS")
    lines.append(f"   Total Animals        : {len(result_df)}")
    lines.append(f"   Average Inbreeding F : {avg_f:.2f}%")
    lines.append(f"   Average EBV          : {avg_ebv:.4f}")

    if "Phenotype" in result_df.columns:
        valid_phenos = pd.to_numeric(result_df["Phenotype"], errors="coerce").dropna()

        if not valid_phenos.empty:
            avg_p = valid_phenos.mean()
            sd_p = valid_phenos.std()
            lines.append(f"   Average Phenotype    : {avg_p:.2f}")
            lines.append(f"   Phenotype SD         : {sd_p:.2f}")

            if settings and "h2" in settings and "intensity" in settings:
                response = calculate_selection_response(settings["h2"], sd_p, settings["intensity"])
                lines.append(f"   Selection Response R : {response:.4f}")

    hwe_res = analyze_hardy_weinberg(result_df)
    lines.append(f"   HWE Genetic Status   : {hwe_res['status']}")
    lines.append("-" * 60)
    lines.append("")

    threshold_ebv = result_df["EBV"].quantile(0.75)
    selection_df = result_df[
        (result_df["EBV"] >= threshold_ebv) &
        (result_df["Inbreeding_%"] < 6.25)
    ].sort_values("EBV", ascending=False).head(10)

    lines.append("2. TOP SELECTION CANDIDATES")
    if not selection_df.empty:
        lines.append(f"   {'Animal_ID':<15} {'EBV':<10} {'F (%)':<8} {'Classification':<25}")
        for _, row in selection_df.iterrows():
            lines.append(
                f"   {str(row['Animal_ID']):<15} {row['EBV']:<10.4f} "
                f"{row['Inbreeding_%']:<8.2f} {str(row['Classification']):<25}"
            )
    else:
        lines.append("   No candidates met the selection criteria.")

    lines.append("-" * 60)
    lines.append("")

    threshold_low_ebv = result_df["EBV"].quantile(0.10)
    culling_df = result_df[
        (result_df["Inbreeding_%"] >= 25) |
        (result_df["Reproduction_Warning"] != "") |
        (result_df["EBV"] <= threshold_low_ebv)
    ].sort_values("Inbreeding_%", ascending=False).head(10)

    lines.append("3. PRIORITY CULLING CANDIDATES")
    if not culling_df.empty:
        lines.append(f"   {'Animal_ID':<15} {'F (%)':<8} {'Warning':<35}")
        for _, row in culling_df.iterrows():
            warning = row["Reproduction_Warning"] if row["Reproduction_Warning"] else "Low genetic value or high F"
            lines.append(
                f"   {str(row['Animal_ID']):<15} {row['Inbreeding_%']:<8.2f} {warning:<35}"
            )
    else:
        lines.append("   No high-priority culling candidates found.")

    lines.append("-" * 60)
    lines.append("")
    lines.append("4. COMPLETE INDIVIDUAL LOG")

    for _, row in result_df.iterrows():
        lines.append(
            f"   [{row['Animal_ID']}] F: {row['Inbreeding_%']:.2f}% | "
            f"EBV: {row['EBV']:.4f} | {row['Classification']}"
        )
        if row["Reproduction_Warning"]:
            lines.append(f"   WARNING: {row['Reproduction_Warning']}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("END OF REPORT")

    return "\n".join(lines)


def generate_pdf(result_df, settings=None):
    if not REPORTLAB_AVAILABLE:
        return None

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    title_style = styles["Heading1"]
    title_style.alignment = 1

    elements.append(Paragraph("LIVESTOCK BREEDING ANALYSIS REPORT", title_style))
    elements.append(Spacer(1, 20))

    avg_f = result_df["Inbreeding_%"].mean()
    avg_ebv = result_df["EBV"].mean()

    elements.append(Paragraph("<b>1. Population Summary</b>", styles["Heading2"]))

    summary_data = [
        ["Total Population", f"{len(result_df)} animals"],
        ["Average Inbreeding F", f"{avg_f:.2f}%"],
        ["Average EBV", f"{avg_ebv:.4f}"],
        ["Analysis Time", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")],
    ]

    if "Phenotype" in result_df.columns:
        valid_phenos = pd.to_numeric(result_df["Phenotype"], errors="coerce").dropna()
        if not valid_phenos.empty:
            avg_p = valid_phenos.mean()
            sd_p = valid_phenos.std()
            summary_data.append(["Average Phenotype", f"{avg_p:.2f}"])
            summary_data.append(["Phenotype SD", f"{sd_p:.2f}"])

            if settings and "h2" in settings and "intensity" in settings:
                response = calculate_selection_response(settings["h2"], sd_p, settings["intensity"])
                summary_data.append(["Selection Response R", f"{response:.4f}"])

    summary_table = Table(summary_data, colWidths=[180, 250])
    summary_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))

    hwe_res = analyze_hardy_weinberg(result_df)

    elements.append(Paragraph("<b>2. Hardy-Weinberg Equilibrium Analysis</b>", styles["Heading2"]))
    elements.append(Paragraph(f"<b>Status:</b> {hwe_res['status']}", styles["Normal"]))
    elements.append(Paragraph(hwe_res["insight"], styles["Normal"]))
    elements.append(Paragraph("<b>Recommendations:</b>", styles["Normal"]))

    for item in hwe_res["saran"]:
        elements.append(Paragraph(f"- {item}", styles["Normal"]))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph("<b>3. Individual Details and Classification</b>", styles["Heading2"]))

    table_data = [["Animal_ID", "F (%)", "EBV", "Classification"]]
    limit_df = result_df.head(1000)

    for _, row in limit_df.iterrows():
        table_data.append([
            str(row["Animal_ID"]),
            f"{row['Inbreeding_%']:.2f}",
            f"{row['EBV']:.4f}",
            str(row["Classification"]),
        ])

    if len(result_df) > 1000:
        table_data.append(["...", "...", "...", "Data truncated"])

    detail_table = Table(table_data, repeatRows=1, colWidths=[110, 80, 80, 180])
    detail_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2563eb")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
    ]))

    elements.append(detail_table)
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("<b>4. Mating Strategy</b>", styles["Heading2"]))
    elements.append(Paragraph(
        "To maintain genetic progress while reducing inbreeding risk, use animals with high EBV and low kinship. "
        "Avoid full-sibling, half-sibling, and parent-offspring mating. Use sire rotation and unrelated semen sources "
        "when needed.",
        styles["Normal"],
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer



def detect_sex_role(animal_id: str, sire_role_ids: set, dam_role_ids: set) -> str:
    """
    Detects animal sex/role using pedigree role and common ID patterns.
    Priority:
    1. If the animal appears as Sire_ID, it is treated as Male/Sire.
    2. If the animal appears as Dam_ID, it is treated as Female/Dam.
    3. If role is not found, use common ID patterns.
    """
    animal_text = str(animal_id).strip()
    animal_upper = animal_text.upper()

    male_keywords = [
        "SIRE", "BULL", "MALE", "PEJANTAN", "JANTAN",
        "M_", "M-", "M.", "L_", "L-", "L."
    ]

    female_keywords = [
        "DAM", "COW", "FEMALE", "INDUK", "BETINA",
        "F_", "F-", "F.", "P_", "P-", "P."
    ]

    in_sire = animal_text in sire_role_ids
    in_dam = animal_text in dam_role_ids

    if in_sire and not in_dam:
        return "Male / Sire Candidate"

    if in_dam and not in_sire:
        return "Female / Dam Candidate"

    if in_sire and in_dam:
        return "Parent Role - Check Sex"

    if any(key in animal_upper for key in male_keywords):
        return "Male / Sire Candidate"

    if any(key in animal_upper for key in female_keywords):
        return "Female / Dam Candidate"

    return "Unidentified Sex"


def add_sex_role_column(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds Sex_Role column to make male and female identification easier.
    """
    df = result_df.copy()

    sire_role_ids = set(
        df["Sire_ID"]
        .dropna()
        .astype(str)
        .loc[lambda s: ~s.str.lower().isin(UNKNOWN_VALUES)]
    )

    dam_role_ids = set(
        df["Dam_ID"]
        .dropna()
        .astype(str)
        .loc[lambda s: ~s.str.lower().isin(UNKNOWN_VALUES)]
    )

    df["Sex_Role"] = df["Animal_ID"].astype(str).apply(
        lambda x: detect_sex_role(x, sire_role_ids, dam_role_ids)
    )

    return df


def infer_mating_candidates(result_df: pd.DataFrame):
    """
    Separates male/sire and female/dam candidates directly.
    If a real Sex column is not available, the system infers sex/role from:
    - whether the animal appears in Sire_ID or Dam_ID,
    - ID patterns such as SIRE, BULL, DAM, COW, JANTAN, BETINA,
    - EBV and inbreeding eligibility.
    """
    df = add_sex_role_column(result_df)

    selectable = df[
        (df["Inbreeding_%"] < 12.5) &
        (~df["Classification"].astype(str).str.contains("Final Stock", case=False, na=False))
    ].copy()

    if selectable.empty:
        selectable = df.copy()

    # Prefer animals with positive EBV when available.
    positive_selectable = selectable[selectable["EBV"] > 0].copy()
    if not positive_selectable.empty:
        selectable = positive_selectable

    sires = selectable[
        selectable["Sex_Role"].astype(str).str.contains("Male|Sire", case=False, na=False)
    ].copy()

    dams = selectable[
        selectable["Sex_Role"].astype(str).str.contains("Female|Dam", case=False, na=False)
    ].copy()

    # Fallback if uploaded data has no clear role/sex information.
    unidentified = selectable[
        selectable["Sex_Role"].astype(str).str.contains("Unidentified|Check", case=False, na=False)
    ].copy()

    if sires.empty and not unidentified.empty:
        sires = unidentified.sort_values(["EBV", "Inbreeding_%"], ascending=[False, True]).head(5).copy()
        sires["Sex_Role"] = "Male / Sire Candidate - Assumed"

    if dams.empty and not unidentified.empty:
        dams = unidentified.sort_values(["EBV", "Inbreeding_%"], ascending=[False, True]).head(10).copy()
        dams["Sex_Role"] = "Female / Dam Candidate - Assumed"

    sires = sires.sort_values(["EBV", "Inbreeding_%"], ascending=[False, True])
    dams = dams.sort_values(["EBV", "Inbreeding_%"], ascending=[False, True])

    return sires, dams

def simulate_mating_pairs(
    result_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    h2_value: float,
    depression_rate_value: float,
    max_offspring_f: float = 6.25,
    max_pairs: int = 20,
) -> pd.DataFrame:
    """
    Simulates recommended mating pairs using the additive relationship matrix.

    Predicted offspring inbreeding:
    F_offspring = 0.5 * relationship(sire, dam) * 100

    Expected offspring EBV:
    EBV_offspring = 0.5 * (EBV_sire + EBV_dam)
    """
    sires, dams = infer_mating_candidates(result_df)

    valid_phenotypes = pd.to_numeric(result_df.get("Phenotype", pd.Series(dtype=float)), errors="coerce").dropna()
    population_avg = valid_phenotypes.mean() if not valid_phenotypes.empty else np.nan

    simulations = []

    for _, sire in sires.iterrows():
        sire_id = str(sire["Animal_ID"])

        for _, dam in dams.iterrows():
            dam_id = str(dam["Animal_ID"])

            if sire_id == dam_id:
                continue

            if sire_id not in matrix_df.index or dam_id not in matrix_df.columns:
                continue

            relationship = float(matrix_df.loc[sire_id, dam_id])
            predicted_f = 0.5 * relationship * 100
            expected_ebv = 0.5 * (float(sire["EBV"]) + float(dam["EBV"]))

            sire_pheno = pd.to_numeric(sire.get("Phenotype", np.nan), errors="coerce")
            dam_pheno = pd.to_numeric(dam.get("Phenotype", np.nan), errors="coerce")

            if not pd.isna(sire_pheno) and not pd.isna(dam_pheno):
                expected_phenotype_base = 0.5 * (float(sire_pheno) + float(dam_pheno))
            elif not pd.isna(population_avg) and h2_value > 0:
                expected_phenotype_base = population_avg + (expected_ebv / h2_value)
            elif not pd.isna(population_avg):
                expected_phenotype_base = population_avg
            else:
                expected_phenotype_base = np.nan

            estimated_depression = calculate_inbreeding_depression(predicted_f, depression_rate_value)

            if not pd.isna(expected_phenotype_base):
                predicted_phenotype_after_depression = expected_phenotype_base - estimated_depression
            else:
                predicted_phenotype_after_depression = np.nan

            if predicted_f <= 0:
                risk_level = "Very Safe"
            elif predicted_f < 6.25:
                risk_level = "Recommended"
            elif predicted_f < 12.5:
                risk_level = "Use with Caution"
            elif predicted_f < 25:
                risk_level = "High Risk"
            else:
                risk_level = "Avoid"

            if predicted_f <= max_offspring_f and expected_ebv > 0:
                decision = "Recommended mating"
            elif predicted_f <= max_offspring_f:
                decision = "Genetically safe, but EBV is not strong"
            elif predicted_f < 12.5 and expected_ebv > 0:
                decision = "Possible, but monitor inbreeding"
            else:
                decision = "Not recommended"

            simulations.append({
                "Suggested_Sire": sire_id,
                "Suggested_Dam": dam_id,
                "Sire_EBV": round(float(sire["EBV"]), 4),
                "Dam_EBV": round(float(dam["EBV"]), 4),
                "Relationship_A": round(relationship, 4),
                "Predicted_Offspring_F_%": round(predicted_f, 4),
                "Expected_Offspring_EBV": round(expected_ebv, 4),
                "Expected_Phenotype_Base": None if pd.isna(expected_phenotype_base) else round(float(expected_phenotype_base), 4),
                "Estimated_Inbreeding_Depression": round(float(estimated_depression), 4),
                "Predicted_Phenotype_After_Depression": None if pd.isna(predicted_phenotype_after_depression) else round(float(predicted_phenotype_after_depression), 4),
                "Risk_Level": risk_level,
                "Decision": decision,
            })

    if not simulations:
        return pd.DataFrame()

    sim_df = pd.DataFrame(simulations)

    sim_df = sim_df.sort_values(
        by=["Predicted_Offspring_F_%", "Expected_Offspring_EBV", "Relationship_A"],
        ascending=[True, False, True],
    )

    return sim_df.head(max_pairs).reset_index(drop=True)





def make_pedigree_html(dot_source: str, title: str = "Pedigree Visualization") -> str:
    """
    Creates a standalone HTML file that renders Graphviz DOT using Viz.js from CDN.
    This version embeds DOT safely using JSON serialization to prevent syntax errors.
    """
    import json

    safe_title = str(title).replace("<", "&lt;").replace(">", "&gt;")
    dot_json = json.dumps(dot_source)

    filename_base = (
        safe_title.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/viz.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/viz.js@2.1.2/full.render.js"></script>
    <style>
        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f8fafc;
            color: #0f172a;
        }}
        header {{
            padding: 18px 24px;
            background: #1e293b;
            color: white;
        }}
        header h1 {{
            margin: 0 0 6px 0;
            font-size: 22px;
        }}
        header p {{
            margin: 0;
            color: #cbd5e1;
            font-size: 14px;
        }}
        main {{
            padding: 20px;
        }}
        .toolbar {{
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 16px;
        }}
        button {{
            border: 0;
            background: #2563eb;
            color: white;
            padding: 10px 14px;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
        }}
        button:hover {{
            background: #1d4ed8;
        }}
        #graph {{
            background: white;
            border: 1px solid #cbd5e1;
            border-radius: 12px;
            padding: 16px;
            overflow: auto;
            min-height: 70vh;
        }}
        #graph svg {{
            max-width: none;
            height: auto;
        }}
        .note {{
            font-size: 13px;
            color: #475569;
            margin-bottom: 12px;
        }}
    </style>
</head>
<body>
    <header>
        <h1>{safe_title}</h1>
        <p>Standalone pedigree visualization. Use browser zoom or scroll to inspect large graphs.</p>
    </header>
    <main>
        <div class="toolbar">
            <button onclick="downloadSvg()">Download SVG</button>
            <button onclick="downloadDot()">Download DOT Source</button>
        </div>
        <div class="note">
            If the graph is very large, rendering may take time. For thousands of nodes, SVG/DOT can also be opened in external graph tools.
        </div>
        <div id="graph">Rendering graph...</div>
    </main>

    <script>
        const dotSource = {dot_json};
        let renderedSvg = "";

        const viz = new Viz();

        viz.renderSVGElement(dotSource)
            .then(function(element) {{
                renderedSvg = new XMLSerializer().serializeToString(element);
                const graph = document.getElementById("graph");
                graph.innerHTML = "";
                graph.appendChild(element);
            }})
            .catch(function(error) {{
                document.getElementById("graph").innerText = "Failed to render graph: " + error + "\\n\\nDOT preview:\\n" + dotSource.slice(0, 1000);
            }});

        function downloadSvg() {{
            if (!renderedSvg) {{
                alert("SVG is not ready yet.");
                return;
            }}
            const blob = new Blob([renderedSvg], {{type: "image/svg+xml"}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "{filename_base}.svg";
            a.click();
            URL.revokeObjectURL(url);
        }}

        function downloadDot() {{
            const blob = new Blob([dotSource], {{type: "text/vnd.graphviz"}});
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = "{filename_base}.dot";
            a.click();
            URL.revokeObjectURL(url);
        }}
    </script>
</body>
</html>
"""

def make_pedigree_report_html(graph_df: pd.DataFrame, dot_source: str, title: str = "Pedigree Report") -> str:
    """
    Creates a readable HTML report that includes graph rendering and a data table.
    """
    table_cols = [
        "Animal_ID", "Sex_Role", "Sire_ID", "Dam_ID", "Phenotype",
        "EBV", "Heterosis", "Inbreeding_%", "Classification",
        "Inbreeding_Condition", "Reproduction_Warning"
    ]
    existing_cols = [col for col in table_cols if col in graph_df.columns]
    table_html = clean_display(graph_df[existing_cols]).to_html(index=False, escape=False)

    graph_html = make_pedigree_html(dot_source, title=title)

    # Inject table before closing main tag.
    table_section = f"""
        <h2>Pedigree Data Table</h2>
        <div style="overflow:auto; background:white; border:1px solid #cbd5e1; border-radius:12px; padding:12px;">
            {table_html}
        </div>
        <style>
            table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
            }}
            th, td {{
                border: 1px solid #cbd5e1;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background: #e2e8f0;
            }}
        </style>
    """
    return graph_html.replace("</main>", table_section + "\n</main>")


def get_family_subgraph(result_df: pd.DataFrame, selected_id: str, generations_up: int = 3, generations_down: int = 2) -> pd.DataFrame:
    """
    Builds a pedigree subgraph around one selected animal:
    - ancestors up to generations_up
    - descendants up to generations_down
    """
    if not selected_id:
        return result_df.head(0).copy()

    df = result_df.copy()
    df["Animal_ID"] = df["Animal_ID"].astype(str)
    df["Sire_ID"] = df["Sire_ID"].apply(lambda x: None if is_unknown(x) else str(x))
    df["Dam_ID"] = df["Dam_ID"].apply(lambda x: None if is_unknown(x) else str(x))

    parents = {
        row["Animal_ID"]: (row["Sire_ID"], row["Dam_ID"])
        for _, row in df.iterrows()
    }

    children = {}
    for _, row in df.iterrows():
        animal = row["Animal_ID"]
        for parent in [row["Sire_ID"], row["Dam_ID"]]:
            if parent:
                children.setdefault(parent, set()).add(animal)

    keep = set([str(selected_id)])

    current = {str(selected_id)}
    for _ in range(generations_up):
        next_level = set()
        for animal in current:
            sire, dam = parents.get(animal, (None, None))
            for parent in [sire, dam]:
                if parent:
                    next_level.add(parent)
        keep.update(next_level)
        current = next_level

    current = {str(selected_id)}
    for _ in range(generations_down):
        next_level = set()
        for animal in current:
            next_level.update(children.get(animal, set()))
        keep.update(next_level)
        current = next_level

    return df[df["Animal_ID"].isin(keep)].copy()


def filter_pedigree_for_visualization(
    result_df: pd.DataFrame,
    mode: str,
    selected_id: Optional[str] = None,
    classification_filter: Optional[list] = None,
    only_inbred: bool = False,
    only_warning: bool = False,
    max_nodes: Optional[int] = None,
    generations_up: int = 3,
    generations_down: int = 2,
) -> pd.DataFrame:
    """
    Filters pedigree data for visualization while keeping full-data access through downloads.
    """
    df = result_df.copy()

    if mode == "Family subgraph around selected animal" and selected_id:
        df = get_family_subgraph(
            df,
            selected_id=selected_id,
            generations_up=generations_up,
            generations_down=generations_down,
        )

    elif mode == "High-risk animals only":
        df = df[
            (df["Inbreeding_%"] > 0) |
            (df["Reproduction_Warning"].astype(str) != "") |
            (df["Classification"].astype(str).str.contains("Final Stock", case=False, na=False))
        ].copy()

    elif mode == "Selection candidates only":
        threshold_ebv = df["EBV"].quantile(0.75)
        df = df[
            (df["EBV"] >= threshold_ebv) &
            (df["Inbreeding_%"] < 6.25)
        ].copy()

    elif mode == "Custom filter":
        if classification_filter:
            df = df[df["Classification"].isin(classification_filter)].copy()

        if only_inbred:
            df = df[df["Inbreeding_%"] > 0].copy()

        if only_warning:
            df = df[df["Reproduction_Warning"].astype(str) != ""].copy()

    if max_nodes is not None and max_nodes > 0 and len(df) > max_nodes:
        df = df.head(max_nodes).copy()

    return df


def make_dot_unlimited(result_df: pd.DataFrame, include_legend: bool = True) -> str:
    """
    Creates a Graphviz DOT pedigree graph without a hard node limit.
    For very large datasets, use download buttons or filtered rendering.
    """
    df = result_df.copy()

    if df.empty:
        return 'digraph Pedigree { label="No data to display"; }'

    df["Animal_ID"] = df["Animal_ID"].astype(str)
    animal_set = set(df["Animal_ID"].astype(str))

    dot = [
        "digraph Pedigree {",
        "rankdir=LR;",
        "splines=true;",
        "overlap=false;",
        'graph [fontname="Arial", fontsize=12, labelloc="t", label="Pedigree Chart Visualization"];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=9];',
        'edge [arrowsize=0.5, fontname="Arial", fontsize=8];',
    ]

    if include_legend:
        dot.extend([
            'subgraph cluster_legend {',
            'label="Legend";',
            'style="rounded,dashed";',
            '"legend_safe" [label="Safe / Not inbred", fillcolor="#FFFFFF"];',
            '"legend_low" [label="Inbred / Low-Moderate Risk", fillcolor="#FFF4CC"];',
            '"legend_high" [label="Very High Inbreeding", fillcolor="#FFE4E1"];',
            '"legend_elite" [label="Elite Stock Border", fillcolor="#FFFFFF", color="#FFD700", penwidth="3.0"];',
            '"legend_breeding" [label="Breeding Stock Border", fillcolor="#FFFFFF", color="#32CD32", penwidth="2.0"];',
            "}",
        ])

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        f_value = float(row.get("Inbreeding_%", 0))
        classification = str(row.get("Classification", ""))
        sex_role = str(row.get("Sex_Role", ""))
        warning = str(row.get("Reproduction_Warning", ""))

        fill = "#FFFFFF"
        border_color = "black"
        penwidth = "1.0"

        if f_value >= 25:
            fill = "#FFE4E1"
        elif f_value > 0:
            fill = "#FFF4CC"

        if classification == "Elite Stock":
            border_color = "#FFD700"
            penwidth = "3.0"
        elif classification == "Breeding Stock":
            border_color = "#32CD32"
            penwidth = "2.0"
        elif "Final Stock" in classification:
            border_color = "#DC2626"
            penwidth = "2.0"

        warning_text = "\\nWARNING" if warning and warning != "-" else ""

        label = (
            f"{row['Animal_ID']}\\n"
            f"{sex_role}\\n"
            f"F={f_value:.2f}% | EBV={float(row.get('EBV', 0)):.2f}\\n"
            f"{classification}{warning_text}"
        )

        dot.append(
            f'"{animal}" [label="{dot_escape(label)}", fillcolor="{fill}", color="{border_color}", penwidth="{penwidth}"];'
        )

    for _, row in df.iterrows():
        animal = dot_escape(row["Animal_ID"])
        sire = row.get("Sire_ID")
        dam = row.get("Dam_ID")

        if not is_unknown(sire) and str(sire) in animal_set:
            dot.append(f'"{dot_escape(sire)}" -> "{animal}" [label="sire", color="#2563EB"];')

        if not is_unknown(dam) and str(dam) in animal_set:
            dot.append(f'"{dot_escape(dam)}" -> "{animal}" [label="dam", color="#DB2777"];')

    dot.append("}")
    return "\n".join(dot)


def pedigree_summary_table(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a compact summary for large pedigree datasets.
    """
    df = result_df.copy()

    return pd.DataFrame({
        "Metric": [
            "Total animals",
            "Animals with known sire",
            "Animals with known dam",
            "Inbred animals",
            "High-risk animals F >= 12.5%",
            "Very high-risk animals F >= 25%",
            "Backcross / reproduction warning",
            "Elite Stock",
            "Breeding Stock",
            "Commercial",
            "Final Stock",
        ],
        "Value": [
            len(df),
            int((~df["Sire_ID"].apply(is_unknown)).sum()),
            int((~df["Dam_ID"].apply(is_unknown)).sum()),
            int((df["Inbreeding_%"] > 0).sum()),
            int((df["Inbreeding_%"] >= 12.5).sum()),
            int((df["Inbreeding_%"] >= 25).sum()),
            int((df["Reproduction_Warning"].astype(str) != "").sum()),
            int((df["Classification"] == "Elite Stock").sum()),
            int((df["Classification"] == "Breeding Stock").sum()),
            int((df["Classification"] == "Commercial").sum()),
            int((df["Classification"].astype(str).str.contains("Final Stock", case=False, na=False)).sum()),
        ],
    })


def build_pair_pool_for_pure_lines(
    result_df: pd.DataFrame,
    matrix_df: pd.DataFrame,
    h2_value: float,
    depression_rate_value: float,
    max_offspring_f: float,
) -> pd.DataFrame:
    """
    Builds all possible safe sire-dam pairings for pure line foundation.
    The safest pairs have low relationship, low predicted offspring F, and positive expected EBV.
    """
    sires, dams = infer_mating_candidates(result_df)

    rows = []

    valid_phenotypes = pd.to_numeric(result_df.get("Phenotype", pd.Series(dtype=float)), errors="coerce").dropna()
    population_avg = valid_phenotypes.mean() if not valid_phenotypes.empty else np.nan

    for _, sire in sires.iterrows():
        sire_id = str(sire["Animal_ID"])

        for _, dam in dams.iterrows():
            dam_id = str(dam["Animal_ID"])

            if sire_id == dam_id:
                continue

            if sire_id not in matrix_df.index or dam_id not in matrix_df.columns:
                continue

            relationship = float(matrix_df.loc[sire_id, dam_id])
            predicted_f = 0.5 * relationship * 100
            expected_ebv = 0.5 * (float(sire["EBV"]) + float(dam["EBV"]))

            sire_pheno = pd.to_numeric(sire.get("Phenotype", np.nan), errors="coerce")
            dam_pheno = pd.to_numeric(dam.get("Phenotype", np.nan), errors="coerce")

            if not pd.isna(sire_pheno) and not pd.isna(dam_pheno):
                expected_phenotype_base = 0.5 * (float(sire_pheno) + float(dam_pheno))
            elif not pd.isna(population_avg) and h2_value > 0:
                expected_phenotype_base = population_avg + (expected_ebv / h2_value)
            elif not pd.isna(population_avg):
                expected_phenotype_base = population_avg
            else:
                expected_phenotype_base = np.nan

            estimated_depression = calculate_inbreeding_depression(predicted_f, depression_rate_value)

            if not pd.isna(expected_phenotype_base):
                predicted_after_depression = expected_phenotype_base - estimated_depression
            else:
                predicted_after_depression = np.nan

            if predicted_f <= 0:
                risk_level = "Very Safe"
            elif predicted_f < 3.125:
                risk_level = "Very Low Risk"
            elif predicted_f <= max_offspring_f:
                risk_level = "Safe"
            elif predicted_f < 12.5:
                risk_level = "Caution"
            else:
                risk_level = "Avoid"

            safety_score = (
                (max_offspring_f - predicted_f) * 2
                + expected_ebv
                - relationship
            )

            rows.append({
                "Sire_ID": sire_id,
                "Sire_Role": sire.get("Sex_Role", "Male / Sire Candidate"),
                "Dam_ID": dam_id,
                "Dam_Role": dam.get("Sex_Role", "Female / Dam Candidate"),
                "Relationship_A": round(relationship, 4),
                "Predicted_F_%": round(predicted_f, 4),
                "Expected_EBV": round(expected_ebv, 4),
                "Expected_Phenotype_Base": None if pd.isna(expected_phenotype_base) else round(float(expected_phenotype_base), 4),
                "Estimated_Depression": round(float(estimated_depression), 4),
                "Predicted_Phenotype_After_Depression": None if pd.isna(predicted_after_depression) else round(float(predicted_after_depression), 4),
                "Risk_Level": risk_level,
                "Safety_Score": round(float(safety_score), 4),
            })

    if not rows:
        return pd.DataFrame()

    pool = pd.DataFrame(rows)
    pool = pool.sort_values(
        ["Predicted_F_%", "Expected_EBV", "Safety_Score"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    return pool


def select_four_safe_pure_lines(
    pair_pool: pd.DataFrame,
    max_offspring_f: float,
    required_lines: int = 4,
) -> pd.DataFrame:
    """
    Selects four foundation lines while trying to avoid repeated sires and dams.
    If the dataset is limited, the system will still provide the best available lines.
    """
    if pair_pool.empty:
        return pd.DataFrame()

    safe_pool = pair_pool[pair_pool["Predicted_F_%"] <= max_offspring_f].copy()

    if safe_pool.empty:
        safe_pool = pair_pool.copy()

    selected = []
    used_sires = set()
    used_dams = set()

    for _, row in safe_pool.iterrows():
        sire = row["Sire_ID"]
        dam = row["Dam_ID"]

        if sire in used_sires or dam in used_dams:
            continue

        selected.append(row)
        used_sires.add(sire)
        used_dams.add(dam)

        if len(selected) >= required_lines:
            break

    if len(selected) < required_lines:
        for _, row in safe_pool.iterrows():
            key = (row["Sire_ID"], row["Dam_ID"])
            existing = {(r["Sire_ID"], r["Dam_ID"]) for r in selected}

            if key in existing:
                continue

            selected.append(row)

            if len(selected) >= required_lines:
                break

    if not selected:
        return pd.DataFrame()

    selected_df = pd.DataFrame(selected).reset_index(drop=True)
    selected_df.insert(0, "Line", [f"Line {chr(65 + i)}" for i in range(len(selected_df))])

    selected_df["GGPS_Male"] = selected_df["Sire_ID"]
    selected_df["GGPS_Female"] = selected_df["Dam_ID"]
    selected_df["GGPS_Expected_F_%"] = selected_df["Predicted_F_%"]
    selected_df["GGPS_Expected_EBV"] = selected_df["Expected_EBV"]

    return selected_df


def simulate_stock_pyramid_from_lines(
    selected_lines: pd.DataFrame,
    max_offspring_f: float,
) -> pd.DataFrame:
    """
    Simulates a 4-level breeding stock pyramid:
    GGPS -> GPS -> PS -> FS.

    Conservative assumption:
    - GPS is produced from selected pure-line GGPS parents.
    - PS is produced from controlled crossing among different lines.
    - FS is final stock from commercial crossing between PS lines.

    This simulation keeps FS as terminal stock and avoids using high-risk combinations.
    """
    if selected_lines.empty:
        return pd.DataFrame()

    lines = selected_lines.reset_index(drop=True).copy()
    rows = []

    # GGPS level: foundation pure line pair
    for _, row in lines.iterrows():
        rows.append({
            "Stage": "GGPS",
            "Line_System": row["Line"],
            "Male_Source": row["GGPS_Male"],
            "Female_Source": row["GGPS_Female"],
            "Breeding_Model": "Pure line foundation",
            "Predicted_Offspring_F_%": row["GGPS_Expected_F_%"],
            "Expected_Offspring_EBV": row["GGPS_Expected_EBV"],
            "Risk_Level": "Safe" if row["GGPS_Expected_F_%"] <= max_offspring_f else "Caution",
            "Output": f"{row['Line']} GGPS replacement candidate",
            "Use": "Maintain pure line nucleus",
        })

    # GPS level: multiplication within selected line, with conservative carry-over.
    for _, row in lines.iterrows():
        gps_f = min(float(row["GGPS_Expected_F_%"]) * 1.10, 100)
        gps_ebv = float(row["GGPS_Expected_EBV"]) * 0.98

        rows.append({
            "Stage": "GPS",
            "Line_System": row["Line"],
            "Male_Source": f"{row['Line']} GGPS male line",
            "Female_Source": f"{row['Line']} GGPS female line",
            "Breeding_Model": "Pure line multiplication",
            "Predicted_Offspring_F_%": round(gps_f, 4),
            "Expected_Offspring_EBV": round(gps_ebv, 4),
            "Risk_Level": "Safe" if gps_f <= max_offspring_f else "Caution",
            "Output": f"{row['Line']} GPS candidate",
            "Use": "Supply parent-stock line",
        })

    # PS level: cross different pure lines in a controlled way.
    # Use four-line structure: A x B and C x D.
    if len(lines) >= 2:
        ps_pairs = []
        if len(lines) >= 4:
            ps_pairs = [(0, 1, "PS Male Line"), (2, 3, "PS Female Line")]
        else:
            ps_pairs = [(0, 1, "PS Composite Line")]

        for i, j, label in ps_pairs:
            left = lines.iloc[i]
            right = lines.iloc[j]
            ps_f = 0.5 * (float(left["GGPS_Expected_F_%"]) + float(right["GGPS_Expected_F_%"])) * 0.50
            ps_ebv = 0.5 * (float(left["GGPS_Expected_EBV"]) + float(right["GGPS_Expected_EBV"]))

            rows.append({
                "Stage": "PS",
                "Line_System": f"{left['Line']} × {right['Line']}",
                "Male_Source": f"{left['Line']} GPS",
                "Female_Source": f"{right['Line']} GPS",
                "Breeding_Model": "Controlled inter-line cross",
                "Predicted_Offspring_F_%": round(ps_f, 4),
                "Expected_Offspring_EBV": round(ps_ebv, 4),
                "Risk_Level": "Safe" if ps_f <= max_offspring_f else "Caution",
                "Output": label,
                "Use": "Produce parent stock for commercial crossing",
            })

    # FS level: terminal four-line commercial cross.
    if len(lines) >= 4:
        line_a = lines.iloc[0]
        line_b = lines.iloc[1]
        line_c = lines.iloc[2]
        line_d = lines.iloc[3]

        ps_male_ebv = 0.5 * (float(line_a["GGPS_Expected_EBV"]) + float(line_b["GGPS_Expected_EBV"]))
        ps_female_ebv = 0.5 * (float(line_c["GGPS_Expected_EBV"]) + float(line_d["GGPS_Expected_EBV"]))
        fs_ebv = 0.5 * (ps_male_ebv + ps_female_ebv)

        ps_male_f = 0.5 * (float(line_a["GGPS_Expected_F_%"]) + float(line_b["GGPS_Expected_F_%"])) * 0.50
        ps_female_f = 0.5 * (float(line_c["GGPS_Expected_F_%"]) + float(line_d["GGPS_Expected_F_%"])) * 0.50
        fs_f = 0.5 * (ps_male_f + ps_female_f) * 0.50

        rows.append({
            "Stage": "FS",
            "Line_System": "(Line A × Line B) × (Line C × Line D)",
            "Male_Source": "PS Male Line from Line A × Line B",
            "Female_Source": "PS Female Line from Line C × Line D",
            "Breeding_Model": "Terminal four-line cross",
            "Predicted_Offspring_F_%": round(fs_f, 4),
            "Expected_Offspring_EBV": round(fs_ebv, 4),
            "Risk_Level": "Safe" if fs_f <= max_offspring_f else "Caution",
            "Output": "Final Stock / Commercial offspring",
            "Use": "Commercial production only, not for breeding nucleus",
        })

    pyramid = pd.DataFrame(rows)
    return pyramid


def make_pure_line_flow_dot(pyramid_df: pd.DataFrame) -> str:
    """
    Creates a simple Graphviz flow for the GGPS-GPS-PS-FS structure.
    """
    dot = [
        "digraph PureLinePyramid {",
        "rankdir=TB;",
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=10, fillcolor="#F8FAFC"];',
        'edge [arrowsize=0.7];',
        '"GGPS" [label="GGPS\\nGreat Grand Parent Stock\\nPure line nucleus", fillcolor="#DBEAFE"];',
        '"GPS" [label="GPS\\nGrand Parent Stock\\nPure line multiplication", fillcolor="#DCFCE7"];',
        '"PS" [label="PS\\nParent Stock\\nControlled inter-line cross", fillcolor="#FEF3C7"];',
        '"FS" [label="FS\\nFinal Stock\\nTerminal commercial offspring", fillcolor="#FEE2E2"];',
        '"GGPS" -> "GPS";',
        '"GPS" -> "PS";',
        '"PS" -> "FS";',
    ]

    if not pyramid_df.empty:
        line_rows = pyramid_df[pyramid_df["Stage"] == "GGPS"]
        for _, row in line_rows.iterrows():
            line_name = str(row["Line_System"])
            safe_line = dot_escape(line_name)
            label = (
                f"{line_name}\\nMale: {row['Male_Source']}\\nFemale: {row['Female_Source']}\\n"
                f"F: {float(row['Predicted_Offspring_F_%']):.2f}%"
            )
            dot.append(
                f'"{safe_line}" [label="{dot_escape(label)}", fillcolor="#FFFFFF"];'
            )
            dot.append(f'"{safe_line}" -> "GGPS";')

    dot.append("}")
    return "\n".join(dot)


def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-main: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f1f5f9;
            --text-sub: #94a3b8;
            --accent-primary: #3b82f6;
            --border-color: #334155;
            --header-gradient: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        }

        @media (prefers-color-scheme: light) {
            :root {
                --bg-main: #f1f5f9;
                --card-bg: #ffffff;
                --text-main: #1e293b;
                --text-sub: #64748b;
                --border-color: #e2e8f0;
            }
        }

        .main {
            background-color: var(--bg-main) !important;
            font-family: 'Inter', sans-serif;
            color: var(--text-main) !important;
        }

        h1, h2, h3, h4, h5, h6, p, span, label {
            color: var(--text-main) !important;
        }

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

        div[data-testid="stMetric"] {
            background: var(--card-bg) !important;
            padding: 1.2rem;
            border-radius: 1rem;
            border: 1px solid var(--border-color) !important;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid var(--border-color) !important;
        }
        </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown("""
        <div class="custom-header">
            <h1 style="font-weight: 800; font-size: 3rem; margin-bottom: 0;">
                BREEDING & INBREEDING ANALYTICS
            </h1>
            <p style="font-size: 1.15rem; font-weight: 400; letter-spacing: 0.04em;">
                A decision support system for livestock genetic evaluation based on pedigree records,
                inbreeding coefficient, Estimated Breeding Value (EBV), heterosis, and genetic relationship analysis.
                <br>
                This application helps breeders, researchers, and farm managers identify superior breeding candidates,
                avoid close-relative mating, reduce inbreeding risk, and design sustainable breeding strategies.
            </p>
        </div>
    """, unsafe_allow_html=True)


def main():
    favicon_path = pathlib.Path(__file__).parent / "assets" / "favicon.svg"

    page_icon = str(favicon_path) if favicon_path.exists() else "🐄"

    st.set_page_config(
        page_title="Breeding & Inbreeding Analytics",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_css()
    render_header()

    with st.sidebar:
        st.markdown("## Configuration")

        mode = st.radio(
            "Select Data Source",
            ["Full cattle example", "Upload own file"],
            help="Use sample data to learn how the system works or upload your own CSV/Excel file.",
        )

        with st.expander("Required Data Format", expanded=False):
            st.markdown("""
            Please make sure your CSV or Excel file contains the following columns:

            - `Animal_ID`: unique livestock identification code.
            - `Sire_ID`: sire or father identification code.
            - `Dam_ID`: dam or mother identification code.
            - `Phenotype` *(optional)*: performance value for EBV calculation, such as body weight, milk yield, growth rate, or fertility score.

            Use `-` if sire or dam information is unknown.

            A more complete pedigree record will produce a more accurate inbreeding coefficient, relationship matrix, and breeding recommendation.
            """)

        if mode == "Full cattle example":
            raw_df = contoh_sapi_lengkap()
        else:
            uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
            if not uploaded:
                st.warning("Please upload a CSV or Excel file to begin.")
                st.stop()

            raw_df = read_file(uploaded)

        if len(raw_df) > 500:
            st.warning(
                f"Large data detected ({len(raw_df)} rows). "
                "The additive relationship matrix may take longer to calculate."
            )

        cols = list(raw_df.columns)

        st.markdown("### Column Mapping")
        id_col = st.selectbox("Animal_ID Column", cols, index=0)
        sire_col = st.selectbox("Sire_ID Column", cols, index=1 if len(cols) > 1 else 0)
        dam_col = st.selectbox("Dam_ID Column", cols, index=2 if len(cols) > 2 else 0)

        phenotype_col = st.selectbox(
            "Phenotype Column - Optional",
            ["-"] + cols,
            index=cols.index("Phenotype") + 1 if "Phenotype" in cols else 0,
            help="Select a phenotype column to calculate Estimated Breeding Value.",
        )

        pheno_val = None if phenotype_col == "-" else phenotype_col

        st.markdown("### Genetic Parameters")
        h2 = st.slider("Heritability (h²)", 0.0, 1.0, 0.3, 0.05)
        depression_rate = st.slider("Inbreeding Depression Rate per 1% F", 0.0, 5.0, 1.0, 0.1)

        st.markdown("### Selection Parameters")
        intensity = st.slider("Selection Intensity (i)", 0.0, 3.0, 1.5, 0.1)

    try:
        internal = standardize_input(raw_df, id_col, sire_col, dam_col, pheno_val)
        std_df, res_df, matrix_df = calculate(
            internal,
            h2=h2,
            depression_rate=depression_rate,
        )

        res_display_data = res_df[res_df["Data_Type"] == "Input data"].copy()
        res_display_data = add_sex_role_column(res_display_data)

        tabs = st.tabs([
            "Results & Analysis",
            "Genetic Visualization",
            "Pedigree Chart",
            "Relationship Matrix",
            "Heterosis & Crossbreeding",
            "Pure Line Pyramid",
        ])

        with tabs[0]:
            st.subheader("Data Summary")

            total_animals = len(res_display_data)
            inbred_animals = len(res_display_data[res_display_data["Inbreeding_%"] > 0])
            avg_f = float(res_display_data["Inbreeding_%"].mean()) if total_animals else 0
            max_f = float(res_display_data["Inbreeding_%"].max()) if total_animals else 0
            avg_ebv = float(res_display_data["EBV"].mean()) if total_animals else 0

            m0, m1, m2, m3, m4 = st.columns(5)
            m0.metric("Heritability h²", f"{h2:.2f}")
            m1.metric("Total Population", f"{total_animals}")
            m2.metric("Inbred Animals", f"{inbred_animals}")
            m3.metric("Average F", f"{avg_f:.2f}%")
            m4.metric("Average EBV", f"{avg_ebv:.4f}")

            st.markdown("### Livestock Classification Distribution")

            if not res_display_data.empty:
                dist = res_display_data["Classification"].value_counts()
                dist_cols = st.columns(max(len(dist), 1))

                for i, (label, count) in enumerate(dist.items()):
                    dist_cols[i].metric(label, f"{count}")

                has_elite = any(res_display_data["Classification"] == "Elite Stock")

                if not has_elite:
                    st.warning("Warning: No Elite Stock was found in this population.")
                    st.markdown("""
                    <div class="info-card">
                    <b>Genetic Improvement Advice:</b><br>
                    The current population does not contain animals classified as <b>Elite Stock</b>, meaning no individual currently meets the ideal combination of high EBV and low inbreeding level.
                    To improve the genetic quality of the next generation, the following actions are recommended:
                    <ol>
                        <li><b>Outcrossing:</b> Introduce unrelated sires or frozen semen from outside the current population.</li>
                        <li><b>Strict Selection:</b> Prioritize animals classified as <b>Breeding Stock</b> as replacement parents.</li>
                        <li><b>Phenotype Re-evaluation:</b> Ensure performance data are recorded accurately because EBV depends strongly on phenotype quality.</li>
                        <li><b>Inbreeding Control:</b> Keep the inbreeding coefficient below 3% whenever possible.</li>
                        <li><b>Sire Rotation:</b> Avoid using one sire too frequently across the population.</li>
                    </ol>
                    </div>
                    """, unsafe_allow_html=True)

            if pheno_val:
                st.markdown("---")
                st.subheader("Selection Response Estimation")

                valid_phenos = pd.to_numeric(res_display_data["Phenotype"], errors="coerce").dropna()

                if not valid_phenos.empty:
                    avg_p = valid_phenos.mean()
                    sd_p = valid_phenos.std()
                    response = calculate_selection_response(h2, sd_p, intensity)

                    valid_heterosis = pd.to_numeric(
                        res_display_data["Heterosis"],
                        errors="coerce",
                    )
                    valid_heterosis = valid_heterosis[valid_heterosis != 0]
                    avg_h = valid_heterosis.mean() if not valid_heterosis.empty else 0.0

                    r1, r2, r3, r4 = st.columns(4)
                    r1.metric("Average Phenotype", f"{avg_p:.2f}")
                    r2.metric("Phenotype SD", f"{sd_p:.2f}")
                    r3.metric("Selection Response R", f"{response:.4f}")
                    r4.metric("Average Heterosis", f"{avg_h:.4f}")

                    st.info(f"""
                    **Result Interpretation**

                    - **Current Population Average:** The average phenotype value is **{avg_p:.2f}** units.
                    - **Predicted Genetic Progress:** With heritability **{h2}** and selection intensity **{intensity}**, the expected improvement in the next generation is **{response:.4f}** units.
                    - **Next Generation Estimate:** The predicted average performance of offspring is approximately **{avg_p + response:.2f}** units.
                    - **Average Heterosis:** The average heterosis value is **{avg_h:.4f}** units.

                    **Scientific Meaning**

                    Selection response estimates how much improvement can be expected after choosing the best animals as parents. Higher selection intensity may increase genetic progress, but it can also increase inbreeding risk if only a few related animals are repeatedly used.

                    **Breeding Recommendation**

                    Use animals with high EBV and low inbreeding as parent candidates. Avoid selecting animals with high EBV if they also have high inbreeding coefficients.
                    """)

                else:
                    st.warning("Phenotype data is invalid or empty, so selection response cannot be calculated.")

            backcross_cases = res_display_data[res_display_data["Reproduction_Warning"] != ""]

            if not backcross_cases.empty:
                st.markdown("---")
                st.error(f"Detected {len(backcross_cases)} sire-daughter mating case(s) or backcross risk.")

                c1, c2 = st.columns(2)

                with c1:
                    st.write("**List of Exposed Individuals:**")
                    st.dataframe(
                        backcross_cases[["Animal_ID", "Sire_ID", "Dam_ID", "Inbreeding_%"]],
                        hide_index=True,
                    )

                with c2:
                    st.markdown("""
                    <div class="info-card" style="border-left: 4px solid #ef4444;">
                    <b>Reproductive Risk Explanation:</b><br>
                    Mating a sire with its own daughter is a high-risk form of close inbreeding and may produce an inbreeding coefficient of approximately <b>25%</b> or higher.
                    <ul>
                        <li><b>Main Risks:</b> reduced fertility, weak immunity, slow growth, birth defects, and low offspring survival.</li>
                        <li><b>AI Strategy:</b> always check livestock records before selecting semen.</li>
                        <li><b>Semen Rotation:</b> use semen from unrelated sires or different genetic lines.</li>
                        <li><b>Management Decision:</b> offspring from close-relative mating should not be prioritized as replacement breeding stock.</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Inbreeding vs Phenotype Relationship Analysis")

            stats = calculate_stats(res_display_data)

            if stats:
                s1, s2, s3 = st.columns(3)

                s1.metric("Correlation r", f"{stats['correlation']:.4f}")
                s2.metric("Regression b", f"{stats['b']:.4f}")
                s3.metric("R-squared", f"{stats['r_squared']:.4f}")

                st.info(
                    f"**Regression Interpretation:** The estimated regression equation is "
                    f"Y = {stats['a']:.2f} + ({stats['b']:.4f})X. "
                    f"This means every 1% increase in inbreeding is predicted to change the phenotype value by "
                    f"{stats['b']:.4f} units. A negative value indicates possible performance decline as inbreeding increases."
                )
            else:
                st.info("Regression analysis requires at least three valid phenotype records.")

            st.markdown("---")
            st.subheader("Hardy-Weinberg Equilibrium Analysis")

            hwe_res = analyze_hardy_weinberg(res_display_data)

            hw1, hw2 = st.columns([0.4, 0.6])

            with hw1:
                st.write(f"#### Status: {hwe_res['status']}")
                st.write(hwe_res["insight"])
                st.caption(
                    "HWE analysis indicates whether the population is close to random mating or already shows signs of accumulated inbreeding."
                )

            with hw2:
                st.write("**Management Strategy Recommendation:**")
                for item in hwe_res["saran"]:
                    st.write(f"- {item}")

            if hwe_res["is_deviating"]:
                st.warning(
                    "The population shows signs of deviation from random mating. This may indicate accumulated inbreeding or repeated use of related animals."
                )
            else:
                st.success(
                    "The population is relatively close to a healthy genetic distribution. Continue pedigree monitoring."
                )

            st.markdown("---")
            st.markdown("### Complete Calculation Results")
            st.caption(
                "This table summarizes sex/role identification, pedigree information, EBV, heterosis, inbreeding coefficient, biological impact, classification, and recommendation."
            )
            st.info(
                "Sex/Role is automatically identified from pedigree usage. Animals appearing as `Sire_ID` are labeled as male/sire candidates, while animals appearing as `Dam_ID` are labeled as female/dam candidates. If the system cannot detect it, the animal will be labeled as unidentified."
            )

            st.dataframe(clean_display(res_display_data), use_container_width=True, height=500)

            st.markdown("---")
            csel1, csel2 = st.columns([0.4, 0.6])

            with csel1:
                st.subheader("Individual Genetic Interpretation")
                selected_animal = st.selectbox("Select Animal:", res_display_data["Animal_ID"])

            if selected_animal:
                row = res_display_data[res_display_data["Animal_ID"] == selected_animal].iloc[0]

                with csel2:
                    st.info(f"**Selected Animal:** {row['Animal_ID']}\n\n{row['Breeding_Interpretation']}")
                    st.markdown(f"""
                    - **Inbreeding Coefficient:** {row['Inbreeding_%']}% ({row['Inbreeding_Condition']})
                    - **Estimated Performance Impact:** -{row['Inbreeding_Depression']} units
                    - **Biological Impact:** {row['Biological_Impact']}
                    - **Recommendation:** {row['Recommendation']}
                    """)

            st.markdown("### Interpretation Guide")

            with st.expander("Click to understand the main breeding analysis terms"):
                st.markdown("""
                **1. Inbreeding Coefficient (F)**  
                Measures the probability that an animal inherits identical genes from both parents due to common ancestry.

                **2. Estimated Breeding Value (EBV)**  
                Estimates the genetic potential of an animal that can be passed to its offspring.

                **3. Inbreeding Depression**  
                The potential reduction in performance caused by increased homozygosity.

                **4. Heterosis**  
                The performance advantage of offspring compared with the average of both parents.

                **5. Selection Response (R)**  
                The expected improvement in the next generation after selecting superior parents.

                **6. Relationship Matrix (A)**  
                A matrix showing genetic relationship values between individuals. Lower values are preferred for mating pairs.
                """)

            st.markdown("### Population Impact Analysis")
            st.info(dampak_inbreeding(max_f))

            st.markdown("### Download Report")
            dl1, dl2, dl3, dl4 = st.columns(4)

            csv_data = clean_display(res_df).to_csv(index=False).encode("utf-8")
            dl1.download_button(
                "Download CSV",
                csv_data,
                "breeding_analytics_data.csv",
                "text/csv",
                use_container_width=True,
            )

            matrix_csv = matrix_df.to_csv().encode("utf-8")
            dl2.download_button(
                "Download Matrix CSV",
                matrix_csv,
                "relationship_matrix.csv",
                "text/csv",
                use_container_width=True,
            )

            current_settings = {"h2": h2, "intensity": intensity}
            txt_report = dots_to_pedigree(res_display_data, settings=current_settings)
            dl3.download_button(
                "Download TXT Summary",
                txt_report.encode("utf-8"),
                "breeding_summary.txt",
                "text/plain",
                use_container_width=True,
            )

            pdf_data = generate_pdf(res_display_data, settings=current_settings)

            if pdf_data:
                dl4.download_button(
                    "Download PDF Report",
                    pdf_data,
                    "breeding_report.pdf",
                    "application/pdf",
                    use_container_width=True,
                )
            else:
                dl4.info("Install reportlab to enable PDF download.")

            st.markdown("---")
            st.subheader("Selection & Culling Recommendations")

            st.markdown("""
            This section identifies animals that should be prioritized as parent candidates and animals that should not be used for future breeding.

            **Selection Candidates** are animals with relatively high EBV and low inbreeding risk.  
            **Culling Candidates** are animals with very high inbreeding, close-relative mating warnings, or low genetic potential.
            """)

            threshold_ebv = res_display_data["EBV"].quantile(0.75)
            selection_df = res_display_data[
                (res_display_data["EBV"] >= threshold_ebv) &
                (res_display_data["Inbreeding_%"] < 6.25)
            ].sort_values("EBV", ascending=False)

            threshold_low_ebv = res_display_data["EBV"].quantile(0.10)
            culling_df = res_display_data[
                (res_display_data["Inbreeding_%"] >= 25) |
                (res_display_data["Reproduction_Warning"] != "") |
                (res_display_data["EBV"] <= threshold_low_ebv)
            ].sort_values("Inbreeding_%", ascending=False)

            rc1, rc2 = st.columns(2)

            with rc1:
                st.success(f"Selection Candidates: {len(selection_df)} animal(s)")
                st.write("Recommended based on high EBV and low inbreeding risk.")
                st.dataframe(
                    selection_df[["Animal_ID", "EBV", "Inbreeding_%", "Classification"]],
                    hide_index=True,
                    use_container_width=True,
                )

            with rc2:
                st.error(f"Culling Candidates: {len(culling_df)} animal(s)")
                st.write("Identified based on high inbreeding, backcross warning, or low genetic value.")
                st.dataframe(
                    culling_df[["Animal_ID", "EBV", "Inbreeding_%", "Reproduction_Warning"]],
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("---")
            st.subheader("Future Mating Strategy for Inbreeding Prevention")

            st.markdown("""
            This section does not only provide general advice. It also identifies which animals are suitable to be mated and simulates the expected offspring result using the additive relationship matrix.

            The main formula used in the simulation is:

            **Predicted Offspring Inbreeding (F) = 0.5 × Relationship between Sire and Dam × 100**

            Therefore, animals with a low relationship value are preferred because they are expected to produce offspring with lower inbreeding risk.
            """)

            selected_sires, selected_dams = infer_mating_candidates(res_display_data)

            ms1, ms2 = st.columns(2)

            with ms1:
                st.markdown("#### Potential Sire Candidates")
                st.caption("Male animals are identified from Sire_ID roles or male ID patterns such as SIRE, BULL, MALE, PEJANTAN, or JANTAN.")
                st.dataframe(
                    selected_sires[["Animal_ID", "Sex_Role", "EBV", "Inbreeding_%", "Classification"]],
                    hide_index=True,
                    use_container_width=True,
                )

            with ms2:
                st.markdown("#### Potential Dam Candidates")
                st.caption("Female animals are identified from Dam_ID roles or female ID patterns such as DAM, COW, FEMALE, INDUK, or BETINA.")
                st.dataframe(
                    selected_dams[["Animal_ID", "Sex_Role", "EBV", "Inbreeding_%", "Classification"]],
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("#### Mating Simulation Settings")

            sim_col1, sim_col2 = st.columns(2)

            with sim_col1:
                max_offspring_f = st.slider(
                    "Maximum acceptable predicted offspring inbreeding F (%)",
                    min_value=0.0,
                    max_value=25.0,
                    value=6.25,
                    step=0.25,
                    help="Pairs with predicted offspring F below this value are considered safer."
                )

            with sim_col2:
                max_sim_pairs = st.slider(
                    "Number of mating pairs to display",
                    min_value=5,
                    max_value=50,
                    value=20,
                    step=5,
                    help="The table will show the best simulated pairs based on low offspring F and high expected EBV."
                )

            simulated_pairs = simulate_mating_pairs(
                res_display_data,
                matrix_df,
                h2_value=h2,
                depression_rate_value=depression_rate,
                max_offspring_f=max_offspring_f,
                max_pairs=max_sim_pairs,
            )

            st.markdown("#### Recommended Mating Pairs and Offspring Simulation")

            if simulated_pairs.empty:
                st.warning(
                    "No suitable mating simulation could be generated. Please check whether the dataset contains enough potential sires and dams, or add a Sex/Role indicator in the animal ID."
                )
            else:
                recommended_only = simulated_pairs[
                    simulated_pairs["Decision"].astype(str).str.contains("Recommended|safe|monitor", case=False, na=False)
                ]

                sim_metric_1, sim_metric_2, sim_metric_3, sim_metric_4 = st.columns(4)

                sim_metric_1.metric("Simulated Pairs", f"{len(simulated_pairs)}")
                sim_metric_2.metric(
                    "Best Offspring F",
                    f"{simulated_pairs['Predicted_Offspring_F_%'].min():.2f}%"
                )
                sim_metric_3.metric(
                    "Highest Expected EBV",
                    f"{simulated_pairs['Expected_Offspring_EBV'].max():.4f}"
                )
                sim_metric_4.metric(
                    "Recommended / Safe Pairs",
                    f"{len(recommended_only)}"
                )

                st.dataframe(
                    simulated_pairs,
                    hide_index=True,
                    use_container_width=True,
                )

                best_pair = simulated_pairs.iloc[0]

                best_sire_id = best_pair["Suggested_Sire"]
                best_dam_id = best_pair["Suggested_Dam"]

                best_sire_role = "Male / Sire Candidate"
                best_dam_role = "Female / Dam Candidate"

                if "Sex_Role" in res_display_data.columns:
                    sire_match = res_display_data[res_display_data["Animal_ID"].astype(str) == str(best_sire_id)]
                    dam_match = res_display_data[res_display_data["Animal_ID"].astype(str) == str(best_dam_id)]

                    if not sire_match.empty:
                        best_sire_role = sire_match.iloc[0]["Sex_Role"]

                    if not dam_match.empty:
                        best_dam_role = dam_match.iloc[0]["Sex_Role"]

                st.success(f"""
                **Best Suggested Pair**

                The most suitable pair based on the current simulation is:

                **Male / Sire:** {best_sire_id}  
                **Female / Dam:** {best_dam_id}

                This means the recommended mating pair is **{best_sire_id} as the male/sire** and **{best_dam_id} as the female/dam**.

                - **Sire role identification:** {best_sire_role}
                - **Dam role identification:** {best_dam_role}
                - **Predicted offspring inbreeding:** {best_pair['Predicted_Offspring_F_%']:.2f}%
                - **Expected offspring EBV:** {best_pair['Expected_Offspring_EBV']:.4f}
                - **Relationship value:** {best_pair['Relationship_A']:.4f}
                - **Estimated inbreeding depression:** {best_pair['Estimated_Inbreeding_Depression']:.4f} units
                - **Risk level:** {best_pair['Risk_Level']}
                - **Decision:** {best_pair['Decision']}
                """)

                st.markdown("#### Simulation Interpretation")

                st.info("""
                **How to read the simulation table:**

                - `Relationship_A` shows the genetic relationship between the proposed sire and dam.
                - `Predicted_Offspring_F_%` estimates the inbreeding coefficient of the future offspring.
                - `Expected_Offspring_EBV` estimates the average genetic merit inherited from both parents.
                - `Expected_Phenotype_Base` estimates offspring performance before inbreeding depression adjustment.
                - `Predicted_Phenotype_After_Depression` estimates performance after subtracting the inbreeding depression effect.
                - `Risk_Level` and `Decision` help determine whether the pair should be used, monitored, or avoided.

                The best mating pair is not always the pair with the highest EBV. A pair with slightly lower EBV but much lower predicted offspring inbreeding is often safer for long-term population improvement.
                """)

                st.download_button(
                    "Download Mating Simulation CSV",
                    simulated_pairs.to_csv(index=False).encode("utf-8"),
                    "mating_simulation.csv",
                    "text/csv",
                    use_container_width=True,
                )

            st.markdown("""
            <div class="info-card" style="border-left: 4px solid #3b82f6;">
            <b>Mating Strategy to Prevent Future Inbreeding:</b>
            <ul>
                <li><b>Avoid close-relative mating:</b> do not mate animals that share the same sire, dam, or close ancestors.</li>
                <li><b>Use unrelated sires:</b> select sires from different genetic lines with low relationship values to the dams.</li>
                <li><b>Apply sire rotation:</b> avoid using one sire too frequently.</li>
                <li><b>Use artificial insemination wisely:</b> choose semen from tested, unrelated sires.</li>
                <li><b>Check the Relationship Matrix:</b> prioritize mating pairs with relationship values close to zero.</li>
                <li><b>Balance EBV and inbreeding:</b> do not select an animal only because it has high EBV.</li>
                <li><b>Evaluate every generation:</b> recalculate EBV, inbreeding, and relationship values whenever new offspring are added.</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("### Important Livestock Breeding Parameters")

            with st.expander("See detailed breeding concept explanation"):
                p1, p2 = st.columns(2)

                with p1:
                    st.markdown("""
                    **1. Heritability (h²)**  
                    Heritability measures how much of the observed phenotype variation is caused by additive genetic factors.

                    - Low heritability (< 0.20): strongly influenced by environment.
                    - Moderate heritability (0.20 - 0.40): common in growth traits.
                    - High heritability (> 0.40): common in carcass or some production traits.

                    **2. Estimated Breeding Value (EBV)**  
                    EBV predicts the genetic merit of an animal as a parent.
                    """)

                with p2:
                    st.markdown("""
                    **3. Selection Intensity (i)**  
                    Selection intensity describes how strict the selection process is.

                    **4. Tandem Selection**  
                    Tandem selection improves one trait first before focusing on another trait.

                    **5. Correlation and Regression**  
                    Correlation shows association strength, while regression estimates phenotype change when inbreeding increases.
                    """)

            st.markdown("### Insights & Management Strategy")

            mi1, mi2 = st.columns(2)

            with mi1:
                st.info("""
                **Why is inbreeding dangerous?**

                Inbreeding increases the chance that harmful recessive genes will appear in offspring. This may lead to:

                - Lower fertility
                - Reduced growth rate
                - Weaker immune system
                - Higher disease susceptibility
                - Lower milk or meat production
                - Higher risk of birth defects
                - Reduced offspring survival
                """)

            with mi2:
                st.success("""
                **Recommended prevention strategy**

                1. Rotate sires regularly.
                2. Use unrelated sires or external semen through artificial insemination.
                3. Avoid full-sibling, half-sibling, and parent-offspring mating.
                4. Record pedigree and phenotype data consistently.
                5. Use this application before making mating decisions.
                6. Recalculate inbreeding and EBV after every new generation.
                """)

        with tabs[1]:
            st.subheader("Genetic Distribution Visualization")

            v1, v2 = st.columns(2)

            with v1:
                st.markdown("### Inbreeding F Distribution")
                f_data = res_display_data["Inbreeding_%"]

                if not f_data.empty:
                    counts, bin_edges = np.histogram(f_data, bins=10)
                    hist_df = pd.DataFrame({
                        "F range (%)": [
                            f"{bin_edges[i]:.1f}-{bin_edges[i + 1]:.1f}"
                            for i in range(len(counts))
                        ],
                        "Animal Count": counts,
                    })
                    st.bar_chart(hist_df.set_index("F range (%)"))

                st.caption("This graph shows the distribution of inbreeding levels in the population.")

            with v2:
                st.markdown("### Top 10 Genetic Potential Based on EBV")
                top_ebv = res_display_data.sort_values("EBV", ascending=False).head(10)

                if not top_ebv.empty:
                    st.bar_chart(top_ebv.set_index("Animal_ID")["EBV"])

                st.caption("This chart shows the top animals with the highest Estimated Breeding Value.")

            st.markdown("---")
            st.markdown("### Inbreeding vs Phenotype Relationship")

            if "Phenotype" in res_display_data.columns:
                scatter_data = res_display_data[
                    res_display_data["Phenotype"].apply(lambda x: not is_unknown(x))
                ].copy()

                if not scatter_data.empty:
                    scatter_data["Phenotype_Val"] = pd.to_numeric(
                        scatter_data["Phenotype"],
                        errors="coerce",
                    )
                    scatter_data["EBV_Size"] = pd.to_numeric(
                        scatter_data["EBV"],
                        errors="coerce",
                    ).abs() + 1

                    st.scatter_chart(
                        scatter_data,
                        x="Inbreeding_%",
                        y="Phenotype_Val",
                        color="Classification",
                        size="EBV_Size",
                    )
                    st.caption(
                        "Dots show the relationship between inbreeding and phenotype. Color indicates classification."
                    )
                else:
                    st.info("No valid phenotype data is available for scatter visualization.")
            else:
                st.info("No phenotype column is available.")

        with tabs[2]:
            st.subheader("Pedigree Chart Visualization")

            st.markdown("""
            This section supports both **complete pedigree information** and **safe visualization for large datasets**. 
            Rendering thousands of nodes directly in the browser is possible, but it may be slow depending on your computer and browser. 
            For large datasets, use filtered modes or download the full DOT file for external visualization.
            """)

            total_nodes = len(res_df)
            total_edges = int((~res_df["Sire_ID"].apply(is_unknown)).sum() + (~res_df["Dam_ID"].apply(is_unknown)).sum())

            pg_m1, pg_m2, pg_m3, pg_m4 = st.columns(4)
            pg_m1.metric("Total Nodes", f"{total_nodes}")
            pg_m2.metric("Estimated Edges", f"{total_edges}")
            pg_m3.metric("Inbred Animals", f"{int((res_df['Inbreeding_%'] > 0).sum())}")
            pg_m4.metric("High Risk F >= 12.5%", f"{int((res_df['Inbreeding_%'] >= 12.5).sum())}")

            with st.expander("Pedigree Summary", expanded=False):
                st.dataframe(
                    pedigree_summary_table(res_df),
                    hide_index=True,
                    use_container_width=True,
                )

            st.markdown("### Visualization Mode")

            graph_mode = st.selectbox(
                "Choose graph display mode",
                [
                    "Family subgraph around selected animal",
                    "High-risk animals only",
                    "Selection candidates only",
                    "Custom filter",
                    "Full pedigree graph",
                ],
                help="For thousands of animals, filtered views are recommended for faster rendering."
            )

            selected_graph_animal = None
            classification_filter = None
            only_inbred_filter = False
            only_warning_filter = False
            generations_up = 3
            generations_down = 2
            render_limit = None

            if graph_mode == "Family subgraph around selected animal":
                selected_graph_animal = st.selectbox(
                    "Select animal to inspect",
                    res_df["Animal_ID"].astype(str).tolist(),
                    help="The graph will show selected animal, ancestors, and descendants."
                )

                fam_col1, fam_col2 = st.columns(2)
                with fam_col1:
                    generations_up = st.slider("Ancestor generations", 1, 10, 3)
                with fam_col2:
                    generations_down = st.slider("Descendant generations", 0, 10, 2)

            elif graph_mode == "Custom filter":
                available_classes = sorted(res_df["Classification"].dropna().astype(str).unique().tolist())

                classification_filter = st.multiselect(
                    "Filter by classification",
                    available_classes,
                    default=available_classes,
                )

                filter_col1, filter_col2 = st.columns(2)
                with filter_col1:
                    only_inbred_filter = st.checkbox("Show only inbred animals", value=False)
                with filter_col2:
                    only_warning_filter = st.checkbox("Show only animals with reproduction warning", value=False)

            elif graph_mode == "Full pedigree graph":
                st.warning(
                    "Full graph mode may be heavy for hundreds or thousands of animals. If rendering is slow, download the DOT file or switch to filtered mode."
                )

            if graph_mode != "Full pedigree graph":
                render_limit = st.number_input(
                    "Maximum nodes to render in browser, 0 means no limit",
                    min_value=0,
                    max_value=10000,
                    value=500,
                    step=50,
                    help="This only limits what is rendered on screen. Full data remains available in downloads."
                )
                render_limit = None if render_limit == 0 else int(render_limit)
            else:
                confirm_full_render = st.checkbox(
                    "I understand the browser may slow down. Render the full graph.",
                    value=False,
                )
                if not confirm_full_render:
                    render_limit = 500
                    st.info("Full rendering is not confirmed, so the preview is limited to the first 500 nodes.")
                else:
                    render_limit = None

            graph_df = filter_pedigree_for_visualization(
                res_df,
                mode=graph_mode,
                selected_id=selected_graph_animal,
                classification_filter=classification_filter,
                only_inbred=only_inbred_filter,
                only_warning=only_warning_filter,
                max_nodes=render_limit,
                generations_up=generations_up,
                generations_down=generations_down,
            )

            st.markdown("### Graph Preview")
            st.caption(
                f"Rendering {len(graph_df)} node(s) in the browser. Full pedigree still contains {len(res_df)} node(s)."
            )

            if graph_df.empty:
                st.warning("No animals match the selected visualization filter.")
            else:
                dot_preview = make_dot_unlimited(graph_df, include_legend=True)
                st.graphviz_chart(dot_preview)

                with st.expander("Rendered Graph Data", expanded=False):
                    display_cols = [
                        "Animal_ID", "Sex_Role", "Sire_ID", "Dam_ID", "Phenotype",
                        "EBV", "Heterosis", "Inbreeding_%", "Classification",
                        "Inbreeding_Condition", "Reproduction_Warning"
                    ]
                    existing_cols = [col for col in display_cols if col in graph_df.columns]
                    st.dataframe(
                        clean_display(graph_df[existing_cols]),
                        hide_index=True,
                        use_container_width=True,
                    )

            st.markdown("### Download Visualization and Pedigree Data")

            full_dot = make_dot_unlimited(res_df, include_legend=True)
            preview_dot = make_dot_unlimited(graph_df, include_legend=True) if not graph_df.empty else ""

            full_html = make_pedigree_report_html(
                res_df,
                full_dot,
                title="Full Pedigree Visualization Report"
            )

            preview_html = make_pedigree_report_html(
                graph_df,
                preview_dot,
                title="Preview Pedigree Visualization Report"
            ) if not graph_df.empty else ""

            export_col1, export_col2, export_col3 = st.columns(3)

            with export_col1:
                st.download_button(
                    "Download Full Visualization HTML",
                    full_html.encode("utf-8"),
                    "full_pedigree_visualization.html",
                    "text/html",
                    use_container_width=True,
                    help="Open this file directly in a browser. It contains the full pedigree graph and data table."
                )

            with export_col2:
                st.download_button(
                    "Download Current View HTML",
                    preview_html.encode("utf-8"),
                    "current_pedigree_view.html",
                    "text/html",
                    use_container_width=True,
                    help="Open this file in a browser. It contains only the currently rendered/filtered view."
                )

            with export_col3:
                pedigree_csv_cols = [
                    "Animal_ID", "Sex_Role", "Sire_ID", "Dam_ID", "Phenotype",
                    "EBV", "Heterosis", "Inbreeding_%", "Classification",
                    "Inbreeding_Condition", "Biological_Impact", "Recommendation",
                    "Reproduction_Warning"
                ]
                existing_export_cols = [col for col in pedigree_csv_cols if col in res_df.columns]
                st.download_button(
                    "Download Full Pedigree CSV",
                    clean_display(res_df[existing_export_cols]).to_csv(index=False).encode("utf-8"),
                    "full_pedigree_information.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with st.expander("Advanced export for graph software", expanded=False):
                st.caption(
                    "Use DOT only if you want to open the graph in Graphviz, Gephi, yEd, or another external graph visualization tool."
                )
                adv_col1, adv_col2 = st.columns(2)

                with adv_col1:
                    st.download_button(
                        "Download Full DOT Source",
                        full_dot.encode("utf-8"),
                        "full_pedigree_graph.dot",
                        "text/vnd.graphviz",
                        use_container_width=True,
                    )

                with adv_col2:
                    st.download_button(
                        "Download Current View DOT Source",
                        preview_dot.encode("utf-8"),
                        "current_pedigree_view.dot",
                        "text/vnd.graphviz",
                        use_container_width=True,
                    )

            st.info("""
            **Recommended workflow for very large samples:**

            1. Use **Family subgraph** to inspect one animal's ancestors and descendants.
            2. Use **High-risk animals only** to quickly identify problematic matings.
            3. Use **Selection candidates only** to inspect animals suitable for breeding.
            4. Use **Full Pedigree DOT** for complete external visualization when the dataset contains thousands of animals.
            """)
        with tabs[3]:
            st.subheader("Additive Relationship Matrix")

            st.write(
                "The relationship matrix shows genetic relationship values between individuals. "
                "For mating decisions, choose pairs with low values or close to zero."
            )

            if len(matrix_df) > 500:
                st.warning(
                    "The matrix is larger than 500x500 and may cause browser lag. Download the CSV if needed."
                )
                if st.button("Display Matrix Anyway"):
                    st.dataframe(matrix_df, use_container_width=True)
            else:
                st.dataframe(matrix_df, use_container_width=True)

        with tabs[4]:
            st.subheader("Heterosis & Crossbreeding Analysis")

            if "Heterosis" in res_display_data.columns:
                heterosis_df = res_display_data.copy()
                heterosis_df["Heterosis"] = pd.to_numeric(
                    heterosis_df["Heterosis"],
                    errors="coerce",
                ).fillna(0)

                avg_heterosis = heterosis_df["Heterosis"].mean()
                max_heterosis = heterosis_df["Heterosis"].max()
                min_heterosis = heterosis_df["Heterosis"].min()

                h1, h2_col, h3 = st.columns(3)
                h1.metric("Average Heterosis", f"{avg_heterosis:.4f}")
                h2_col.metric("Highest Heterosis", f"{max_heterosis:.4f}")
                h3.metric("Lowest Heterosis", f"{min_heterosis:.4f}")

                st.info("""
                **Heterosis Interpretation**

                Heterosis measures offspring performance compared with the average performance of both parents.
                A positive value indicates that the offspring performs better than the parental average.
                A negative value indicates that the offspring performs below the parental average.

                In commercial livestock systems, heterosis is useful for improving growth, survival, fertility, and production traits through planned crossbreeding.
                """)

                top_h = heterosis_df.sort_values("Heterosis", ascending=False).head(10)

                st.markdown("### Top Heterosis Individuals")
                st.dataframe(
                    top_h[["Animal_ID", "Sire_ID", "Dam_ID", "Phenotype", "Heterosis", "EBV", "Inbreeding_%"]],
                    hide_index=True,
                    use_container_width=True,
                )

                if not top_h.empty:
                    st.bar_chart(top_h.set_index("Animal_ID")["Heterosis"])

                st.markdown("### Crossbreeding Management Notes")
                st.markdown("""
                - Positive heterosis may indicate useful genetic combinations between sire and dam.
                - Avoid using heterosis alone as a selection criterion; always check EBV and inbreeding level.
                - Use crossbreeding to restore genetic diversity when inbreeding is increasing.
                - For purebred improvement, use crossbreeding carefully because it may change breed composition.
                """)
            else:
                st.info("Heterosis data is not available.")


        with tabs[5]:
            st.subheader("Pure Line Simulation for GGPS, GPS, PS, and FS")

            st.markdown("""
            This module simulates a safe **four-line breeding pyramid** to support the development of:

            - **GGPS**: Great Grand Parent Stock as the pure-line nucleus.
            - **GPS**: Grand Parent Stock as pure-line multiplication stock.
            - **PS**: Parent Stock from controlled inter-line crossing.
            - **FS**: Final Stock as terminal commercial offspring.

            The goal is to keep the pure lines safe by choosing male and female founders with low relationship values, low predicted offspring inbreeding, and acceptable EBV.
            """)

            st.warning("""
            Important: this is a decision-support simulation. A real pure-line program should ideally use many males and females per line, not only one pair. If the uploaded dataset has limited animals, the system will still show the safest available structure, but it should be validated by a breeding expert before implementation.
            """)

            pl_col1, pl_col2, pl_col3 = st.columns(3)

            with pl_col1:
                pure_max_f = st.slider(
                    "Maximum safe F for pure-line offspring (%)",
                    min_value=0.0,
                    max_value=12.5,
                    value=6.25,
                    step=0.25,
                    help="Lower values are safer for long-term pure-line development."
                )

            with pl_col2:
                required_lines = st.slider(
                    "Number of pure lines",
                    min_value=2,
                    max_value=4,
                    value=4,
                    step=1,
                    help="Four lines are recommended to build a GGPS-GPS-PS-FS pyramid."
                )

            with pl_col3:
                min_founder_note = st.selectbox(
                    "Program objective",
                    [
                        "Balanced safety and EBV",
                        "Prioritize lowest inbreeding",
                        "Prioritize higher EBV with safe F",
                    ],
                )

            pair_pool = build_pair_pool_for_pure_lines(
                res_display_data,
                matrix_df,
                h2_value=h2,
                depression_rate_value=depression_rate,
                max_offspring_f=pure_max_f,
            )

            selected_lines = select_four_safe_pure_lines(
                pair_pool,
                max_offspring_f=pure_max_f,
                required_lines=required_lines,
            )

            if pair_pool.empty or selected_lines.empty:
                st.error(
                    "The system could not build a pure-line simulation from the current data. Please provide more animals with clear sire/dam information or add more unrelated male and female candidates."
                )
            else:
                st.markdown("### 1. Selected GGPS Pure-Line Founders")

                founder_display = selected_lines[[
                    "Line",
                    "GGPS_Male",
                    "Sire_Role",
                    "GGPS_Female",
                    "Dam_Role",
                    "Relationship_A",
                    "GGPS_Expected_F_%",
                    "GGPS_Expected_EBV",
                    "Risk_Level",
                ]].copy()

                founder_display = founder_display.rename(columns={
                    "GGPS_Male": "GGPS_Male_Sire",
                    "GGPS_Female": "GGPS_Female_Dam",
                    "GGPS_Expected_F_%": "Predicted_GGPS_Offspring_F_%",
                    "GGPS_Expected_EBV": "Expected_GGPS_Offspring_EBV",
                })

                st.dataframe(
                    founder_display,
                    hide_index=True,
                    use_container_width=True,
                )

                safe_count = len(selected_lines[selected_lines["GGPS_Expected_F_%"] <= pure_max_f])
                unique_sires = selected_lines["GGPS_Male"].nunique()
                unique_dams = selected_lines["GGPS_Female"].nunique()

                pm1, pm2, pm3, pm4 = st.columns(4)
                pm1.metric("Selected Lines", f"{len(selected_lines)}")
                pm2.metric("Safe Lines", f"{safe_count}")
                pm3.metric("Unique Males", f"{unique_sires}")
                pm4.metric("Unique Females", f"{unique_dams}")

                if len(selected_lines) < 4:
                    st.warning(
                        "Less than four lines could be selected from the current data. Add more unrelated male and female candidates to build a complete four-line GGPS-GPS-PS-FS structure."
                    )

                if unique_sires < len(selected_lines) or unique_dams < len(selected_lines):
                    st.warning(
                        "Some sires or dams are reused across lines. For a safer pure-line program, each line should ideally have different unrelated male and female founders."
                    )

                st.markdown("### 2. GGPS → GPS → PS → FS Pyramid Simulation")

                pyramid_df = simulate_stock_pyramid_from_lines(
                    selected_lines,
                    max_offspring_f=pure_max_f,
                )

                st.dataframe(
                    pyramid_df,
                    hide_index=True,
                    use_container_width=True,
                )

                st.markdown("### 3. Breeding Pyramid Flow")

                st.graphviz_chart(make_pure_line_flow_dot(pyramid_df))

                st.markdown("### 4. Recommended Four-Line Structure")

                if len(selected_lines) >= 4:
                    line_a = selected_lines.iloc[0]
                    line_b = selected_lines.iloc[1]
                    line_c = selected_lines.iloc[2]
                    line_d = selected_lines.iloc[3]

                    st.success(f"""
                    **Safe Four-Line Pyramid Recommendation**

                    **GGPS Pure Lines**
                    - **Line A:** Male/Sire `{line_a['GGPS_Male']}` × Female/Dam `{line_a['GGPS_Female']}`
                    - **Line B:** Male/Sire `{line_b['GGPS_Male']}` × Female/Dam `{line_b['GGPS_Female']}`
                    - **Line C:** Male/Sire `{line_c['GGPS_Male']}` × Female/Dam `{line_c['GGPS_Female']}`
                    - **Line D:** Male/Sire `{line_d['GGPS_Male']}` × Female/Dam `{line_d['GGPS_Female']}`

                    **GPS Stage**
                    - Maintain each line separately as pure-line multiplication stock.

                    **PS Stage**
                    - Produce **PS Male Line** from `Line A × Line B`.
                    - Produce **PS Female Line** from `Line C × Line D`.

                    **FS Stage**
                    - Produce commercial **Final Stock** from `(Line A × Line B) × (Line C × Line D)`.

                    This structure helps preserve pure-line identity at the GGPS/GPS level while reducing inbreeding risk at PS and FS levels through controlled inter-line crossing.
                    """)
                else:
                    st.info(
                        "A complete four-line recommendation requires four selected lines. The current simulation shows the safest partial structure available from the uploaded dataset."
                    )

                st.markdown("### 5. Safety Rules for Pure-Line Production")

                st.info("""
                **Recommended safety rules:**

                1. Keep **GGPS and GPS** as pure-line nucleus and multiplication stock.
                2. Do not use **FS** as breeding stock. FS should be terminal commercial stock.
                3. Avoid mating animals with predicted offspring F above the selected safety threshold.
                4. Maintain more than one male and one female per line whenever possible.
                5. Use the relationship matrix before every mating decision.
                6. Recalculate the pyramid every generation because relationship values and inbreeding risks will change.
                7. If a line repeatedly produces high F values, introduce unrelated animals into that line or redesign the line structure.
                """)

                st.download_button(
                    "Download Pure Line Founder Plan CSV",
                    founder_display.to_csv(index=False).encode("utf-8"),
                    "pure_line_founder_plan.csv",
                    "text/csv",
                    use_container_width=True,
                )

                st.download_button(
                    "Download GGPS-GPS-PS-FS Pyramid Simulation CSV",
                    pyramid_df.to_csv(index=False).encode("utf-8"),
                    "pure_line_pyramid_simulation.csv",
                    "text/csv",
                    use_container_width=True,
                )

    except Exception as e:
        st.error("An error occurred while processing the data.")
        st.exception(e)
        st.info("""
        Please check the following:

        1. `Animal_ID`, `Sire_ID`, and `Dam_ID` columns are mapped correctly.
        2. Animal IDs are unique and consistent.
        3. Missing parents are written as `-` or left empty.
        4. The pedigree does not contain cycles, for example an animal listed as its own ancestor.
        5. Phenotype values are numeric if used for EBV calculation.
        """)


if __name__ == "__main__":
    main()
